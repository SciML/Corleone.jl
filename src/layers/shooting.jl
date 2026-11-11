"""
$(TYPEDEF)

A Lux layer that solves a differential equation in segments separated by parameter
switches and initial-condition resets. Calling with `(input, ps, st)` returns
`(trajectory, next_st)`, with reset residuals stored in `trajectory.residuals`.
A tunable assignment at the global problem start is a free initial value and
produces no matching residual. Active tunable interior assignments produce matching
residuals; fixed resets do not. Assignments at the final time create no outgoing segment.
A differential equation problem can be supplied as `input`; other inputs use the
stored problem. A `(problem, tstops)` tuple supplies an explicit segment grid.

Segment boundaries are computed once during setup and stored in `st.timestops`.
Forward calls reuse that vector and preserve it in the returned state. After
changing a schedule, explicitly refresh the cached grid before solving; see
[`get_timestops`](@ref). For a replacement problem with a different time span,
supply its matching grid through the `(problem, tstops)` call.

`EnsembleSerial()` solves sequentially. For ODE problems, `EnsembleThreads()` and
`EnsembleDistributed()` solve independent blocks starting at interior full-state
resets. Partial resets remain sequential within a block. Reverse-mode rules
record block solves serially; ordinary forward execution uses the ensemble map.

# Fields
$(FIELDS)
"""
@concrete terse struct ShootingLayer <: LuxCore.AbstractLuxContainerLayer{(:initialcondition, :parameter)}
    "The optional name of the shooting layer."
    name
    "The differential equation problem used when no replacement problem is supplied."
    problem
    "The differential equation solver algorithm for each segment."
    alg
    "The ensemble algorithm selecting serial, threaded, or distributed block execution."
    ensemblealg
    "Per-variable initial-condition layers and their reset schedules."
    initialcondition
    "Per-parameter layers supplying parameter values for each segment."
    parameter
    "The ControlSymbolCache shared with trajectories for symbolic bookkeeping."
    sys
end

# Resolve symbolic metadata once, outside the forward/AD path. Keep the existing
# positional constructor for callers that replace the problem or sublayers.
ShootingLayer(name, problem, alg, ensemblealg, initialcondition, parameter) =
    ShootingLayer(name, problem, alg, ensemblealg, initialcondition, parameter, ControlSymbolCache(problem))

SymbolicIndexingInterface.symbolic_container(layer::ShootingLayer) = layer.sys

# Retain effective annotation bounds, which would otherwise be discarded after
# constructing the numerical sublayers. Explicit annotations override defaults.
# Explicit transforms own their domain and leave the natural metadata unchanged.
function shooting_metadata(problem, definitions, ic_defs; controls, quadratures, bounds)
    parent = ControlSymbolCache(problem)
    cache = ControlSymbolCache(
        parent;
        controls = isnothing(controls) ? parent.controls : controls,
        quadratures = isnothing(quadratures) ? parent.quadratures : quadratures,
        bounds = isnothing(bounds) ? parent.bounds : merge(parent.bounds, bounds)
    )
    effective_bounds = nothing
    for (i, definition) in enumerate(definitions)
        annotation = definition isa Pair ? first(definition) : definition
        annotation isa AbstractAnnotation || continue
        has_lowerbound(annotation) || has_upperbound(annotation) || continue
        kind = i in ic_defs ? Val(:variable) : Val(:parameter)
        for scalar in scalar_annotations(annotation)
            get_transform(scalar) isa BoundedTransform || continue
            resolved = update_annotation(set_kind(scalar, kind), cache)
            isnothing(effective_bounds) && (effective_bounds = Dict{Any, Any}(cache.bounds))
            effective_bounds[get_sym(resolved)] = get_bounds(resolved)
        end
    end
    return isnothing(effective_bounds) ? cache : ControlSymbolCache(cache; bounds = effective_bounds)
end


"""
$(SIGNATURES)

Construct a shooting layer from a problem, solver, and `annotation => timegrid`
definitions. Variable annotations define initial-condition resets; remaining
definitions configure parameters. `fixed_parameters` selects parameters held in
Lux state.

`controls` and `quadratures` select parameter and state subsets by symbol or original
index. `bounds` maps symbols to bound pairs used as annotation defaults. Omitted
metadata inherits from the problem cache; explicit annotation bounds take precedence.
Metadata does not alter numerical state or parameter layouts.
"""
function ShootingLayer(
        problem::SciMLBase.AbstractSciMLProblem,
        alg::SciMLBase.AbstractSciMLAlgorithm,
        definitions...;
        name = nothing,
        fixed_parameters = [],
        ensemblealg = EnsembleSerial(),
        controls = nothing,
        quadratures = nothing,
        bounds = nothing,
    )
    # ReverseDiff treats bottom-typed empty arrays as tracked broadcast inputs.
    definitions = isempty(definitions) ? Any[] : collect(definitions)
    ic_defs = findall(definitions) do definition
        annotation = definition isa Pair ? first(definition) : definition
        annotation isa AbstractAnnotation && is_variable(annotation)
    end
    p_defs = setdiff(eachindex(definitions), ic_defs)
    sys = shooting_metadata(problem, definitions, ic_defs; controls, quadratures, bounds)
    ics = InitialConditions(problem, definitions[ic_defs]...; sys)
    params = Parameter(problem, definitions[p_defs]...; fixed = fixed_parameters, sys)
    return ShootingLayer(name, problem, alg, ensemblealg, ics, params, sys)
end

"""
$(SIGNATURES)

Return sorted, unique segment boundaries from the problem time span and the
initial-condition and parameter schedules in Lux state. Layer boundaries outside
the problem time span are excluded. The result has element type `T`. Schedules
are read exclusively from Lux state, so `ps` may be `nothing`. To refresh a layer's
cached grid after changing schedules, use
`merge(st, (; timestops = get_timestops(layer, nothing, st)))`. Forward evaluation
uses `st.timestops` directly rather than invoking this function.
"""
get_timestops(layer::ShootingLayer, ps, st, T = eltype(layer.problem.tspan)) =
    shooting_timestops(layer.initialcondition, layer.parameter, layer.problem, st, T)

function shooting_timestops(initialcondition, parameter, problem, st, T)
    tstops = vcat(
        collect(T, problem.tspan),
        get_timestops(initialcondition, nothing, st.initialcondition, T),
        get_timestops(parameter, nothing, st.parameter, T)
    )
    filter!(t -> first(problem.tspan) <= t <= last(problem.tspan), tstops)
    sort!(tstops)
    return unique!(tstops)
end

# Visit only the Lux children: the problem and symbolic cache are metadata,
# and must not participate in generic container field inference during setup.
function LuxCore.initialparameters(rng::Random.AbstractRNG, layer::ShootingLayer)
    return (;
        initialcondition = LuxCore.initialparameters(rng, layer.initialcondition),
        parameter = LuxCore.initialparameters(rng, layer.parameter),
    )
end

function LuxCore.initialstates(rng::Random.AbstractRNG, layer::ShootingLayer)
    st = (;
        initialcondition = LuxCore.initialstates(rng, layer.initialcondition),
        parameter = LuxCore.initialstates(rng, layer.parameter),
    )
    tstops = get_timestops(layer, nothing, st)
    return merge(st, (; timestops = tstops))
end

(x::ShootingLayer)(::Any, ps, st) = x(x.problem, ps, st)
# Feeding a state vector starts a stage from an explicit initial condition (used by
# multistage chains to hand off the previous stage's final state).
(x::ShootingLayer)(u0::AbstractArray, ps, st) = x(remake(x.problem; u0 = u0), ps, st)

@inline function (x::ShootingLayer)(prob::SciMLBase.AbstractSciMLProblem, ps, st::NamedTuple)
    return x((prob, st.timestops), ps, st)
end

@inline function (x::ShootingLayer)((prob, tstops)::Tuple, ps, st::NamedTuple)
    # Explicit grids become the returned state's cache, just like setup grids.
    state = merge(st, (; timestops = tstops))
    solutions, residuals, next_st = solve_problems(x.ensemblealg, x, prob, ps, state, tstops)
    # Distributed results contain serialized metadata; retain the caller's cache.
    next_st = merge(next_st, (; timestops = tstops))
    return build_trajectory(x, ps, next_st, solutions, residuals), next_st
end

solve_problems(alg, ::ShootingLayer, prob, ps, st, tstops) = throw(ArgumentError("Unsupported shooting algorithm $(alg) for $(typeof(prob)). Parallel shooting currently supports ODE problems with EnsembleThreads() or EnsembleDistributed()."))

@inline function solve_problems(::EnsembleSerial, x::ShootingLayer, prob::SciMLBase.AbstractSciMLProblem, ps, st, tstops)
    t0, tinf = tstops[1:2]
    saveat = get(st, :saveat, nothing)
    save_options = isnothing(saveat) ? (;) : (; saveat)
    (u0, variable_residuals), st_ic = x.initialcondition(
        (prob.u0, t0), ps.initialcondition, st.initialcondition;
        matching = t0 != first(prob.tspan)
    )
    p, st_p = x.parameter(t0, ps.parameter, st.parameter)
    # Put the effective inputs in the problem before solve dispatch so scalar-type
    # changes (e.g. Float64 to Dual) retain an inferable solution type.
    tspan = sampling_tspan((t0, tinf), saveat)
    segment = remake(prob; u0 = u0, p = p, tspan)
    sol = solve(segment, x.alg; save_everystep = false, dense = false, save_start = true, save_end = true, save_options...)
    sol = pad_failed_segment!(sol, t0, tinf, get(save_options, :saveat, ()))
    next_st = merge(st, (; initialcondition = st_ic, parameter = st_p))
    # Parameters can first introduce AD scalars in the solve, after the initial
    # (possibly empty) residual vector has already been constructed.
    residual_type = promote_type(eltype(variable_residuals), eltype(terminal_state(sol)))
    residuals = residual_type === eltype(variable_residuals) ? variable_residuals : residual_type.(variable_residuals)
    return buffered_solve(x, prob, sol, residuals, ps, next_st, tstops[2:end], save_options)
end

sampling_tspan(tspan, ::Nothing) = tspan
function sampling_tspan(tspan, samples)
    T = promote_type(typeof(first(tspan)), eltype(samples))
    return (convert(T, first(tspan)), convert(T, last(tspan)))
end

get_du0(::Any, ::Any) = nothing
function get_du0(prob::SciMLBase.AbstractDAEProblem, sol)
    isnothing(sol.du) || return sol.du[end]
    # DFBDF can omit derivative history. The next segment then starts from the
    # problem's du0, which only an initialization that recomputes it makes consistent.
    recomputes_du0(prob) || throw(
        ArgumentError(
            "DAE segment handoff without saved derivatives needs an initialization that recomputes du0; " *
                "set e.g. initializealg = BrownFullBasicInit() on the DAEProblem."
        )
    )
    return nothing
end
function recomputes_du0(prob)
    alg = get(prob.kwargs, :initializealg, nothing)
    alg isa Union{SciMLBase.NoInit, SciMLBase.CheckInit} && return false
    # The default only recomputes through symbolic initialization data.
    return !isnothing(alg) || SciMLBase.has_initialization_data(prob.f)
end

function internal_solve(prob::SciMLBase.AbstractSciMLProblem, alg::SciMLBase.AbstractSciMLAlgorithm, sol::SciMLBase.AbstractSciMLSolution, u0, p, tspan, save_options)
    du0 = get_du0(prob, sol)
    span = sampling_tspan(tspan, get(save_options, :saveat, nothing))
    return internal_solve_with_du0(prob, alg, u0, p, span, du0, save_options)
end

# Keep both sides of segment boundaries; reduce continuous handoffs when assembling the trajectory.
internal_solve_with_du0(prob, alg, u0, p, tspan, ::Nothing, save_options) =
    solve(prob, alg; u0, p, tspan, save_everystep = false, dense = false, save_start = true, save_end = true, save_options...)
internal_solve_with_du0(prob, alg, u0, p, tspan, du0, save_options) =
    solve(prob, alg; u0, du0, p, tspan, save_everystep = false, dense = false, save_start = true, save_end = true, save_options...)

# A failed segment (e.g. DAE initialization failure) may save only its initial
# point; an integrator that exhausted its iterations still saves its last computed
# step. Fill every expected saved time the failed segment did not reach, repeating
# that last computed state, so the assembled trajectory keeps the prepared time
# grid and the optimizer sees a removable infeasibility instead of a compiled
# expression indexing past the end of a short trajectory.
#
# `expected` lists additional expected saved times (e.g. experiment saveat),
# which may span multiple segments. The saved time grid must be compatible with the caller's
# prepared grid; the pad repeats the last computed state verbatim, so only missing
# times are appended.
function pad_failed_segment!(sol, t0, t1, expected = ())
    if !SciMLBase.successful_retcode(getproperty(sol, :retcode)) && sol.t isa Vector
        for t in pad_missing_times(sol, t1, expected)
            push!(sol.t, t)
            push!(sol.u, last(sol.u))
        end
    end
    return sol
end

function pad_missing_times(sol, t1, expected)
    t_end = last(sol.t)
    missing = sort!(collect(t for t in expected if t_end < t < t1))
    t_end < t1 && push!(missing, t1)
    return missing
end

# Zygote traces `push!` into the solution arrays and rejects array mutation, so
# gradient handling for the pad cannot be traced through; it needs an explicit
# rule. The forward pass repeats the last computed state, so a padded state entry
# is a copy of `sol.u[n]` and a padded time entry is the corresponding argument
# value. The cotangent formats follow SciMLBase: solution cotangents are
# `RecursiveArrayTools.DiffEqArray`s or `Tangent`s with `:u`/`:t` fields. The
# unknown-format fallback zeroes the failure point's contribution, keeping AD
# working (the pad itself carries no derivative information; repeating an already
# computed state is the only choice available).
function ChainRulesCore.rrule(::typeof(pad_failed_segment!), sol, t0, t1, expected)
    n_pre = sol.t isa Vector ? length(sol.t) : 0
    padded = pad_failed_segment!(sol, t0, t1, expected)
    n_pushed = sol.t isa Vector ? length(sol.t) - n_pre : 0
    function pad_failed_segment_pullback(Δsol)
        if n_pushed == 0 || Δsol isa AbstractZero
            # No pad applied or no information flowing in; identity or zeros.
            return ChainRulesCore.NoTangent(), Δsol, ChainRulesCore.ZeroTangent(), ChainRulesCore.ZeroTangent(), ChainRulesCore.ZeroTangent()
        end
        if Δsol isa RecursiveArrayTools.AbstractVectorOfArray
            Δu, Δt = Δsol.u, Δsol.t
            rebuild = (u, t) -> RecursiveArrayTools.DiffEqArray(u, t)
        elseif Δsol isa ChainRulesCore.Tangent
            # Partial struct cotangents: missing fields read as ZeroTangent.
            Δu, Δt = Δsol.u, Δsol.t
            # NamedTuple primals take a plain NamedTuple cotangent (constructing a
            # `Tangent` of a NamedTuple with partial fields is unsupported); other
            # structs rebuild a full `Tangent` with the missing fields zeroed.
            if sol isa NamedTuple
                rebuild = (u, t) -> (; u = u, t = t)
            else
                rebuild = (u, t) -> ChainRulesCore.Tangent{typeof(sol)}(; u = u, t = t)
            end
        else
            return ChainRulesCore.NoTangent(), ChainRulesCore.ZeroTangent(), ChainRulesCore.ZeroTangent(), ChainRulesCore.ZeroTangent(), ChainRulesCore.ZeroTangent()
        end
        # State cotangents of the pad repeats accumulate into the repeated source
        # state; time cotangents of the pad repeats belong to the padded-in
        # argument times (t1 last), while the original time cotangents pass on.
        n = length(sol.t) - n_pushed
        t1_cot = ChainRulesCore.ZeroTangent()
        t_cot = Δt
        if !(Δt isa AbstractZero)
            t1_cot = Δt[n + n_pushed]
            t_cot = collect(Δt[i] for i in 1:n)
        end
        u_cot = Δu
        if !(Δu isa AbstractZero)
            u_cot = collect(Δu[i] for i in 1:n)
            for i in (n + 1):(n + n_pushed)
                u_cot[n] = u_cot[n] + Δu[i]
            end
        end
        return ChainRulesCore.NoTangent(), rebuild(u_cot, t_cot), ChainRulesCore.ZeroTangent(), t1_cot, ChainRulesCore.ZeroTangent()
    end
    return padded, pad_failed_segment_pullback
end


function mkbuffer(::T, vec::AbstractArray) where {T}
    buffer = similar(vec, T)
    return is_extension_loaded(Val{:Zygote}()) ? wrapbuffer(buffer) : buffer
end

function wrapbuffer end # Implemented by the Zygote extension.

unwrapbuffer(x) = x

function buffered_solve(x::ShootingLayer, prob, sol::T, res::R, ps, st, tstops, save_options) where {T, R}
    solbuff = mkbuffer(sol, tstops)
    solbuff[1] = sol
    resbuff = mkbuffer(res, tstops)
    resbuff[1] = res
    return buffered_solve!(solbuff, x, prob, resbuff, ps, st, tstops, save_options)
end

"""
$(SIGNATURES)

Return the final state from a time-series solution, a sequence of saved states,
or a matrix whose columns are saved states. Supports solver and AD return formats.
"""
terminal_state(sol::SciMLBase.AbstractTimeseriesSolution) = last(sol.u)
terminal_state(sol::Base.AbstractVecOrTuple) = last(sol)
terminal_state(sol::AbstractArray) = sol[:, end]

@inline function buffered_solve!(sols, x::ShootingLayer, prob, res, ps, st, tstops, save_options)
    @inbounds for i in Base.OneTo(length(tstops) - 1)
        (u0, res[i + 1]), st_ic = x.initialcondition((terminal_state(sols[i]), tstops[i]), ps.initialcondition, st.initialcondition)
        p, st_p = x.parameter(tstops[i], ps.parameter, st.parameter)
        tspan = (tstops[i], tstops[i + 1])
        solution = internal_solve(prob, x.alg, sols[i], u0, p, tspan, save_options)
        sols[i + 1] = pad_failed_segment!(solution, tspan..., get(save_options, :saveat, ()))
        st = merge(st, (; initialcondition = st_ic, parameter = st_p))
    end
    return unwrapbuffer(sols), unwrapbuffer(res), st
end

get_lower_bound(layer::ShootingLayer, ps, st = nothing) =
    shooting_parameter_bounds(get_lower_bound, first, layer, ps, st)
get_upper_bound(layer::ShootingLayer, ps, st = nothing) =
    shooting_parameter_bounds(get_upper_bound, last, layer, ps, st)

function shooting_parameter_bounds(operation, select, layer, ps, st)
    # Build the parameter tree through the same generic Lux recursion. Applying
    # symbolic metadata is the only ShootingLayer-specific part of this protocol.
    bounds = container_parameter_bounds(operation, layer, ps, st, (:initialcondition, :parameter))
    variable_bounds = symbolic_parameter_bounds(
        bounds.initialcondition.initialconditions,
        layer.sys, variable_symbols(layer.sys), select, layer.initialcondition.initialconditions
    )
    parameter_bounds = symbolic_parameter_bounds(
        bounds.parameter.parameters,
        layer.sys, parameter_symbols(layer.sys), select, layer.parameter.parameters
    )
    return merge(
        bounds, (;
            initialcondition = merge(bounds.initialcondition, (; initialconditions = variable_bounds)),
            parameter = merge(bounds.parameter, (; parameters = parameter_bounds)),
        )
    )
end

function symbolic_parameter_bounds(template::NamedTuple, cache, symbols, select, layers)
    names = keys(template)
    fields = map(names) do name
        child = layers[name]
        if child isa Union{InitialCondition, PiecewiseConstant} && child.parameter isa NamedTuple
            inner_names = keys(template[name].parameter)
            inner = NamedTuple{inner_names}(
                map(inner_names) do variable
                    sym = only(filter(sym -> Symbol(sym) == variable, symbols))
                    bounds = internal_bounds(collection_transform(child.parameter[variable]), get_bounds(cache, sym))
                    parameter_bound_tree(template[name].parameter[variable], select(bounds))
                end
            )
            return merge(template[name], (; parameter = inner))
        end
        sym = only(filter(sym -> Symbol(sym) == name, symbols))
        parameter_bound_tree(template[name], select(internal_bounds(collection_transform(child), get_bounds(cache, sym))))
    end
    return NamedTuple{names}(fields)
end

"""
$(SIGNATURES)

Count scalar state and parameter matching residuals for the cached segment grid in
`st.timestops`, without running the model. Each tunable initial condition contributes
one residual at each matching segment start strictly after the global problem
start. Assignments at the global start are free initial values. Each inserted
interval of a tunable parameter adds one matching residual,
counted once regardless of other segment boundaries within that interval. Original
parameter switches, fixed assignments, and resets at the final endpoint contribute none.

The count uses scheduling metadata in Lux state; `ps` may be `nothing`. Refresh
`st.timestops` after changing schedules, then recompute this count before allocating
optimizer constraint buffers. This discrete metadata calculation is not differentiated.
"""
get_number_of_shooting_constraints(layer::ShootingLayer, ps, st) =
    shooting_constraint_count(layer.initialcondition, layer.parameter, layer.problem, st)

function shooting_constraint_count(initialcondition, parameter, problem, st)
    n = 0
    for name in keys(initialcondition.initialconditions)
        ic = initialcondition.initialconditions[name]
        ic isa InitialCondition || continue
        state = st.initialcondition.initialconditions[name]
        for i in firstindex(st.timestops):(lastindex(st.timestops) - 1)
            st.timestops[i] == first(problem.tspan) && continue
            n += resets_variable(ic, state, st.timestops[i]) * number_of_tunable_collections(ic.parameter)
        end
    end
    return n + number_of_parameter_constraints(parameter, st.parameter)
end

ChainRulesCore.@non_differentiable get_number_of_shooting_constraints(layer, ps, st)

"""
$(SIGNATURES)

Split tunable parameter intervals at interior full-state resets and return updated
`(ps, st)` for the same layer. New interval values duplicate the previously active
value. Their right-hand interval indices are recorded in `st`'s per-parameter
`lookup.inserted` vectors, for subsequent parameter matching constraints.

Existing parameter switches remain unconstrained, and fixed parameters are unchanged.
Repeated calls do not insert additional entries. The input arrays and layer are
retained unchanged; changed parameter arrays and lookup grids are newly allocated,
and affected lookup caches are cleared. Segment boundaries are read from the current
initial-condition schedules and the stored problem span; the returned state's cached
timestops are refreshed. No solve, initialization, or constraint construction occurs.

The layer's setup dimensions remain unchanged. Use the returned parameters to size
optimizer storage and query bounds; calling setup again restores the original layout.
"""
function add_parameter_segments!(layer::ShootingLayer, ps, st)
    tstops = get_timestops(layer, nothing, st)
    blocks = shooting_ranges(layer, st, tstops)
    times = [tstops[first(block)] for block in blocks[2:end]]
    isempty(times) && return ps, merge(st, (; timestops = tstops))
    parameters, state = add_parameter_segments!(layer.parameter, ps.parameter, st.parameter, times)
    next_ps = replace_parameter_fields(ps, (; parameter = parameters))
    next_st = merge(st, (; parameter = state, timestops = tstops))
    return next_ps, next_st
end

optimization_grid(layer::ShootingLayer, st, timepoints) = expression_evaluation_grid(layer, st, timepoints)
optimization_schedule(::ShootingLayer, st) = st.timestops
optimization_state(::ShootingLayer, st, grid) = merge(st, (; timestops = grid))
optimization_prediction(layer::ShootingLayer, ps, st, grid) = first(layer((layer.problem, grid), ps, st))

function trajectory_objective(layer::ShootingLayer, loss)
    if isnothing(loss)
        symbols = Tuple(quadrature_symbols(layer.sys))
        isempty(symbols) && throw(ArgumentError("Supply an objective or declare quadrature states."))
        return sum ∘ getsym(layer.sys, symbols) ∘ terminal_optimization_sample
    elseif loss isa Union{Symbol, Integer}
        return getsym(layer.sys, loss) ∘ terminal_optimization_sample
    end
    return loss
end

# Arbitrary dynamics may couple all states. Propagate their combined dependency
# set through each integration segment, retaining precision at scheduled resets.
function shooting_constraint_dependencies(layer::ShootingLayer, ids, st; sample_times = nothing, hessian = false)
    layer.problem isa SciMLBase.AbstractODEProblem || throw(ArgumentError("Sparse shooting dependencies currently require an ODEProblem."))
    dependencies = [BitSet() for _ in 1:sum(length, layer.initialcondition.indices; init = 0)]
    rows = BitSet[]
    reset_rows = Vector{Union{Nothing, BitSet}}(nothing, length(dependencies))
    reset_hessian_rows = hessian ? similar(reset_rows) : nothing
    matching_hessian = BitSet[]
    states_at = Dict{Real, Vector{BitSet}}()
    # Every saved segment endpoint conservatively couples all states, so whole
    # state histories have the same union. It is only read after this walk.
    history = BitSet()
    state_series = fill(history, length(dependencies))
    callback = get(layer.problem.kwargs, :callback, nothing)
    has_callbacks = callback isa SciMLBase.CallbackSet ?
        !isempty(callback.continuous_callbacks) || !isempty(callback.discrete_callbacks) : !isnothing(callback)
    f = layer.problem.f
    initializes_state = has_callbacks || !hasproperty(f, :initialization_data) ||
        !isnothing(f.initialization_data) || !hasproperty(f, :mass_matrix) || f.mass_matrix != LinearAlgebra.I
    for (segment, t) in enumerate(st.timestops[1:(end - 1)])
        fill!(reset_rows, nothing)
        hessian && fill!(reset_hessian_rows, nothing)
        for (k, name) in enumerate(keys(layer.initialcondition.initialconditions))
            reset_dependency!(
                dependencies, reset_rows, layer.initialcondition.initialconditions[name],
                getproperty(ids.initialcondition.initialconditions, name), st.initialcondition.initialconditions[name],
                layer.initialcondition.indices[k], t, t != first(layer.problem.tspan); hessian_rows = reset_hessian_rows
            )
        end
        for i in eachindex(reset_rows)
            isnothing(reset_rows[i]) && continue
            push!(rows, reset_rows[i])
            hessian && push!(matching_hessian, reset_hessian_rows[i])
        end
        combined = active_parameter_columns(layer.parameter, ids.parameter, st.parameter, t)
        previous = nothing
        for dependency in dependencies
            if dependency !== previous
                union!(combined, dependency)
                previous = dependency
            end
        end
        query = isnothing(sample_times) ? 0 : searchsortedfirst(sample_times, t)
        if query > 0 && query <= length(sample_times) && sample_times[query] == t
            # A callback initializer can mix states and parameters before the
            # first saved value. Keep that boundary conservative when present.
            outgoing = !initializes_state ?
                copy(dependencies) : fill(combined, length(dependencies))
            states_at[t] = outgoing
        end
        # Published dependency sets are read-only. Resets replace vector entries,
        # so coupled states can share one set without copying its bit storage.
        dependencies = fill(combined, length(dependencies))
        stop = st.timestops[segment + 1]
        if !isnothing(sample_times)
            first_sample = searchsortedlast(sample_times, t) + 1
            last_sample = searchsortedlast(sample_times, stop)
            if first_sample <= last_sample
                snapshot = copy(dependencies)
                for i in first_sample:last_sample
                    states_at[sample_times[i]] = snapshot
                end
            end
            # This superset includes both sides of every saved reset.
            union!(history, combined)
        end
    end
    for entry in parameter_constraint_entries(layer.parameter, ids.parameter, st.parameter)
        entry.fixed && continue
        for i in entry.inserted
            push!(rows, BitSet([Int(entry.values[i - 1]), Int(entry.values[i])]))
            hessian && push!(matching_hessian, BitSet()) # Affine parameter equality.
        end
    end
    return (; matching = rows, states_at, state_series, matching_hessian)
end
