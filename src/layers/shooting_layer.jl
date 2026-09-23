"""
$(TYPEDEF)

Runs multiple shooting intervals (each a `ShootingInterval`) concurrently, solving
piecewise-constant ODE segments within each interval via `sequential_solve`. The degree
of parallelism is controlled by `ensemble_algorithm` (default: `EnsembleSerial()`).

When called with a `DEProblem`, returns a tuple of `ShootingSegment`s (one per
shooting interval) and updated states.

# Fields
$(FIELDS)
"""
@concrete terse struct ShootingLayer <: LuxCore.AbstractLuxContainerLayer{(:intervals, :controls)}
    "The (controlled) symbolic system cache"
    sys
    "The `ShootingInterval`s"
    intervals
    "The controls acting on the system"
    controls
    "The algorithm to solve the layer with"
    algorithm
    "The ensemble algorithm to control parallelism"
    ensemble_algorithm
end


function ShootingLayer(
        problem::SciMLBase.AbstractDEProblem,
        variable_id,
        controls...;
        shooting_method::AbstractAutoShoot = NoShoot(),
        algorithm::SciMLBase.AbstractDEAlgorithm,
        ensemble_algorithm::SciMLBase.EnsembleAlgorithm = EnsembleSerial(),
        tspan = problem.tspan,
        kwargs...
    )
    controlnames = reduce(vcat, map(Base.Fix2(getfield, :parameter_id), controls))
    cache = ControlSymbolCache(problem, collect(controlnames), get(kwargs, :quadratures, []))
    emptyv = empty(variable_symbols(cache))
    controls = Controls(controls...; sys = something(Solutions.get_symbolic_container(problem.f), Solutions.default_cache(problem)))
    reset!(controls)
    timepoints = get(problem.kwargs, :saveat, eltype(tspan)[])
    append!(timepoints, collect(tspan))
    unique!(sort!(timepoints))
    shooting_points = optimal_shooting_points(shooting_method, controls, LuxCore.setup(Random.default_rng(), controls)...; timepoints)
    append!(shooting_points, collect(tspan))
    unique!(sort!(shooting_points))
    ics = map(enumerate(zip(shooting_points[1:(end - 1)], shooting_points[2:end]))) do (i, tspan)
        ShootingInterval(
            problem, i == 1 ? eltype(emptyv).(variable_id) : minimal_variable_symbols(cache), tspan;
            controls = i == 1 ? emptyv : get_shooted_controls(controls, tspan),
            get(kwargs, :shooting_intervals, (;))...
        )
    end
    return ShootingLayer(
        cache, tuple(ics...), controls, algorithm, ensemble_algorithm
    )
end

# For evaluation
mythreadmap(::EnsembleSerial, args...) = map(args...)
mythreadmap(::EnsembleThreads, f, args::Tuple...) = begin
    # Not NTuple{N}...: N and the tuple eltype T can't both be bound from the
    # argument types alone (an empty NTuple{0,T} is the same type Tuple{} for any
    # T), which Aqua's unbound-args check correctly flags. N is unused for dispatch
    # here, only for reconstructing the output length, so compute it directly.
    res = tmap(f, collect.(args)...)
    N = length(first(args))
    ntuple(i -> res[i], N)
end
mythreadmap(::EnsembleThreads, f, args...) = tmap(f, args...)
mythreadmap(::EnsembleDistributed, args...) = pmap(args...)

function sequential_solve(cache, prob::ODEProblem, alg, setter, controls, ps, st, tspans::AbstractVector)
    (t0, t1) = first(tspans)
    p, st = controls(t0, ps, st)
    sol = solve(prob, alg; p = setter(p), tspan = (t0, t1), save_everystep = false, save_start = true, save_end = true)
    ret = Solutions.ControlSegment(sol, cache)
    length(tspans) == 1 && return vcat(ret)
    new_prob = remake(sol.prob, u0 = sol.u[end])
    return vcat(ret, sequential_solve(cache, new_prob, alg, setter, controls, ps, st, tspans[2:end]))
end

function sequential_solve(cache, prob::DAEProblem, alg, setter, controls, ps, st, tspans::AbstractVector)
    (t0, t1) = first(tspans)
    p, st = controls(t0, ps, st)
    _prob = remake(prob, p = setter(p), tspan = (t0, t1))
    integ = init(_prob, alg, save_everystep = false, save_start = true, save_end = true)
    sol = solve!(integ)
    ret = Solutions.ControlSegment(sol, cache)
    length(tspans) == 1 && return vcat(ret)
    new_prob = remake(sol.prob, u0 = integ.u, du0 = integ.du)
    return vcat(ret, sequential_solve(cache, new_prob, alg, setter, controls, ps, st, tspans[2:end]))
end

@generated function get_probs(ic::NTuple{N, Any}, controls, prob, ps, st) where {N}
    probs = [gensym() for _ in Base.OneTo(N)]
    tspans = [gensym() for _ in Base.OneTo(N)]
    sts = [gensym() for _ in Base.OneTo(N)]
    exprs = Expr[]
    for i in Base.OneTo(N)
        push!(
            exprs,
            :(($(probs[i]), $(sts[i])) = ic[$(i)](prob, ps.intervals[$(i)], st.intervals[$(i)]))
        )
        push!(
            exprs,
            :($(tspans[i]) = collect_timegrid(controls, ps.controls, st.controls, $(probs[i]).tspan))
        )
    end
    push!(exprs, :(return ($(Expr(:tuple, probs...)), $(Expr(:tuple, tspans...)), $(Expr(:tuple, sts...)))))
    return Expr(:block, exprs...)
end

in_tspan((ti, _)::Tuple, (t0, tinf)::Tuple) = t0 <= ti < tinf

function solve_segment(args)
    out = sequential_solve(Base.front(args)...)
    shooting_vars = Base.last(args)
    Solutions.ShootingSegment(out, first(args), shooting_vars)
end

function (layer::ShootingLayer)(problem::SciMLBase.AbstractDEProblem, ps, st)
    (; sys, intervals, controls, algorithm, ensemble_algorithm) = layer
    probs, tgrids, st_interval = get_probs(intervals, controls, problem, ps, st)
    setter = let p0 = problem.p
        (ps) -> begin
            SciMLStructures.replace(SciMLStructures.Tunable(), p0, ps)
        end
    end
    args = ntuple(
        i -> (
            sys, probs[i],
            algorithm, setter, controls, ps.controls, st.controls,
            tgrids[i], get_shooting_variables(intervals[i], sys)
        ), length(intervals)
    )
    sols = mythreadmap(ensemble_algorithm, solve_segment, args)
    return Trajectory(sols, sys), merge(st, (; interval = st_interval))
end

# Overrides the generic AbstractLuxContainerLayer recursion (which would sum
# PiecewiseParameter-injected breakpoints across :controls): the constraints a
# ShootingLayer actually contributes once solved are the state-continuity
# constraints computed by Solutions.Trajectory.shooting_constraints — one per
# non-quadrature state, per gap between intervals — independent of which
# variables happen to be tunable on each ShootingInterval.
function get_number_of_shooting_constraints(layer::ShootingLayer{N}) where N 
    N == 1 && return 0 
    return sum(2:N) do i 
        get_number_of_shooting_constraints(layer.intervals[i])
    end
end

function collect_timegrid(layer::ShootingLayer, ps, st)
    (; controls, intervals) = layer 
    tspans = reduce(vcat, map(intervals) do interval 
        collect(interval.tspan)
    end)
    tgrid = collect_timegrid(controls, ps.controls, st.controls, extrema(tspans))
    append!(tspans, reduce(vcat, map(collect, tgrid)))
    sort!(tspans)
    unique!(tspans)
    tspans
end
