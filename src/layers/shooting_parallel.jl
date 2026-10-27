# Scheduling depends on exact grid membership, including fixed initial conditions;
# the residual activity mask deliberately excludes those fixed conditions.
resets_variable(::PassThrough, st, t) = false
resets_variable(ic::InitialCondition, st, t) = haskey(ic.lookup, t, nothing, st.lookup)
resets_variable(ic::FixedLayer, st, t) = resets_variable(ic.layer, st.states, t)

function shooting_ranges(x::ShootingLayer, st, tstops)
    ics = x.initialcondition.initialconditions
    starts = [firstindex(tstops)]
    for i in (firstindex(tstops) + 1):(lastindex(tstops) - 1)
        # Only state owns the grids. Do not iterate parameter values here:
        # ComponentArray.values flattens its numeric storage rather than fields.
        resets = map(values(ics), values(st.initialcondition.initialconditions)) do ic, state
            resets_variable(ic, state, tstops[i])
        end
        all(resets) && push!(starts, i)
    end
    return [
        starts[i]:(i == length(starts) ? lastindex(tstops) : starts[i + 1])
            for i in eachindex(starts)
    ]
end
@non_differentiable shooting_ranges(::Any, ::Any, ::Any)

"""
$(SIGNATURES)

Return one state per independent shooting interval, with its local `timestops`
and private mutable scheduling buffers. Adjacent intervals share their boundary
timepoint. Numerical values stored in state retain their references and AD paths;
the same `ps` can be used for every interval.
"""
function shooting_states(layer::ShootingLayer, ps, st, tstops)
    ranges = shooting_ranges(layer, st, tstops)
    return map(ranges) do indices
        merge(private_shooting_state(layer, st), (; timestops = tstops[indices]))
    end
end

# Copy scheduling buffers only. In particular, FixedLayer.parameters and all
# differentiable numeric values in Lux state retain their original references.
copy_shooting_cache(x) = copy(x)
copy_shooting_cache(::Nothing) = nothing
@non_differentiable copy_shooting_cache(::Any)

private_shooting_state(::Union{PassThrough, ParameterCollection}, st) = st
private_shooting_state(::LookUp, st) = merge(
    st, (;
        inserted = copy_shooting_cache(st.inserted), cache = copy_shooting_cache(st.cache),
    )
)
private_shooting_state(ic::FixedLayer, st) = (;
    states = private_shooting_state(ic.layer, st.states), parameters = st.parameters,
)
private_shooting_state(ic::Union{InitialCondition, PiecewiseConstant}, st) =
    merge(st, (; lookup = private_shooting_state(ic.lookup, st.lookup)))
function private_shooting_state(layer::InitialConditions, st)
    return (;
        initialconditions = map(private_shooting_state, layer.initialconditions, st.initialconditions),
        active = copy_shooting_cache(st.active),
    )
end
function private_shooting_state(layer::Parameter, st)
    # Paired NamedTuple mapping preserves each heterogeneous child's field type.
    return (; parameters = map(private_shooting_state, layer.parameters, st.parameters))
end
private_shooting_state(x::ShootingLayer, st) = merge(
    st, (;
        initialcondition = private_shooting_state(x.initialcondition, st.initialcondition),
        parameter = private_shooting_state(x.parameter, st.parameter),
    )
)

shooting_map(f, alg, blocks) = mythreadmap(alg, f, blocks)

# Record and execute reverse-mode solves serially until task-local AD contexts
# and cotangent reduction are available. This uses one forward solve per block;
# ordinary forward execution (including ForwardDiff) still uses the ensemble map.
function ChainRulesCore.rrule(
        config::RuleConfig{>:HasReverseMode},
        ::typeof(shooting_map), f, alg, blocks
    )
    output, pullback = rrule_via_ad(config, map, f, blocks)
    function shooting_map_pullback(delta)
        _, df, dblocks = pullback(delta)
        return NoTangent(), df, NoTangent(), dblocks
    end
    return output, shooting_map_pullback
end

# Extract once so the solution, residual and state paths do not accumulate a mix
# of native tuple cotangents and ProjectTo-generated Tangent{Tuple}s in Zygote.
unpack_shooting_results(results) = (map(first, results), map(r -> r[2], results), last(results)[3])

function ChainRulesCore.rrule(::typeof(unpack_shooting_results), results)
    output = unpack_shooting_results(results)
    function unpack_results_pullback(delta)
        delta = unthunk(delta)
        delta isa AbstractZero && return NoTangent(), ZeroTangent()
        dsolutions, dresiduals, dst = unthunk(delta[1]), unthunk(delta[2]), unthunk(delta[3])
        dresults = map(eachindex(results)) do i
            ds = dsolutions isa AbstractZero ? ZeroTangent() : dsolutions[i]
            dr = dresiduals isa AbstractZero ? ZeroTangent() : dresiduals[i]
            (ds, dr, i == lastindex(results) ? dst : ZeroTangent())
        end
        return NoTangent(), dresults
    end
    return output, unpack_results_pullback
end

function solve_problems(
        ensemblealg::Union{EnsembleThreads, EnsembleDistributed},
        x::ShootingLayer, prob::SciMLBase.AbstractODEProblem, ps, st, tstops
    )
    states = shooting_states(x, ps, st, tstops)
    # No independent restart: retain the serial solver and its buffers.
    length(states) == 1 && return solve_problems(EnsembleSerial(), x, prob, ps, only(states), only(states).timestops)
    results = shooting_map(ensemblealg, states) do local_st
        solve_problems(EnsembleSerial(), x, prob, ps, local_st, local_st.timestops)
    end
    solution_blocks, residual_blocks, next_st = unpack_shooting_results(results)
    solutions = reduce(vcat, solution_blocks)
    # A block can solve from its fully assigned start without its predecessor.
    # Its first residual cannot: reconnect that edge after all solves complete.
    residuals = map(eachindex(residual_blocks)) do i
        block_residuals = residual_blocks[i]
        i == 1 && return block_residuals
        incoming = terminal_state(last(solution_blocks[i - 1]))
        (_, boundary_residual), _ = x.initialcondition(
            (incoming, first(states[i].timestops)),
            ps.initialcondition, private_shooting_state(x.initialcondition, states[i].initialcondition)
        )
        vcat([boundary_residual], block_residuals[2:end])
    end
    return solutions, reduce(vcat, residuals), next_st
end
