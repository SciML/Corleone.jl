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
    controls = Controls(controls...; sys = something(Solutions.get_symbolic_container(problem.f), Solutions.default_cache(problem)))
    reset!(controls)
    timepoints = get(problem.kwargs, :saveat, eltype(tspan)[])
    append!(timepoints, collect(tspan))
    unique!(sort!(timepoints))
    shooting_points = optimal_shooting_points(shooting_method, controls, LuxCore.setup(Random.default_rng(), controls)...; timepoints)
    append!(shooting_points, collect(tspan))
    unique!(sort!(shooting_points))
    ics = map(enumerate(zip(shooting_points[1:end-1], shooting_points[2:end]))) do (i,tspan)
        ShootingInterval(problem, i == 1 ? variable_id : variable_symbols(cache), tspan; 
            get(kwargs, :shooting_intervals, (;))...
        )
    end
    ShootingLayer(
        cache, tuple(ics...), controls, algorithm, ensemble_algorithm
    )
end

# For evaluation
mythreadmap(::EnsembleSerial, args...) = map(args...)
mythreadmap(::EnsembleThreads, args...) = tmap(args...)
mythreadmap(::EnsembleDistributed, args...) = pmap(args...)

function sequential_solve(cache, prob, alg, setter, controls, ps, st, tspans::AbstractVector)
    (t0, t1) = first(tspans)
    p, st = controls(t0, ps, st)
    sol = solve(prob, alg; p = setter(p), tspan = (t0, t1), save_everystep = false)
    ret = Solutions.ControlSegment(sol, cache)
    length(tspans) == 1 && return vcat(ret)
    new_prob = remake(sol.prob, u0 = sol.u[end])
    return vcat(ret, sequential_solve(cache, new_prob, alg, setter, controls, ps, st, tspans[2:end]))
end

@generated function get_probs(ic::NTuple{N, Any}, controls, prob, ps, st) where N
    probs = [gensym() for _ in Base.OneTo(N)]
    tspans = [gensym() for _ in Base.OneTo(N)]
    sts = [gensym() for _ in Base.OneTo(N)]
    exprs = Expr[]
    for i in Base.OneTo(N)
        push!(exprs, 
            :(($(probs[i]), $(sts[i])) = ic[$(i)](prob, ps.intervals[$(i)], st.intervals[$(i)]))
        ) 
        push!(exprs, 
            :($(tspans[i]) = collect_timegrid(controls, ps.controls, st.controls, $(probs[i]).tspan))
        )
    end
    push!(exprs, :(return ($(Expr(:tuple, probs...)), $(Expr(:tuple, tspans...)), $(Expr(:tuple, sts...)))) )
    return Expr(:block, exprs...)
end

in_tspan((ti, _)::Tuple, (t0, tinf)::Tuple) = t0 <= ti < tinf

function (layer::ShootingLayer)(problem::SciMLBase.AbstractDEProblem, ps, st)
    (; sys, intervals, controls, algorithm, ensemble_algorithm) = layer 
    probs, tgrids, st_interval = get_probs(intervals, controls, problem, ps, st)
    setter = let p0 = problem.p
        (ps) -> begin 
            SciMLStructures.replace(SciMLStructures.Tunable(), p0, ps)
        end
    end
    args = ntuple(i->(sys, probs[i], 
        algorithm, setter, controls, ps.controls, st.controls, 
        tgrids[i]), length(intervals))
    sols = mythreadmap(ensemble_algorithm, Base.Fix2(Solutions.ShootingSegment, control_cache) ∘ Base.splat(sequential_solve), args)
    Trajectory(sols, sys), merge(st, (; interval = st_interval))
end

