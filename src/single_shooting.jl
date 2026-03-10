struct SingleShootingLayer{A, U0, C} <: LuxCore.AbstractLuxContainerLayer{(:initial_conditions, :controls)}
    "The name of the container"
    name::Symbol
    "The algorithm to solve the underlying DEProblem"
    algorithm::A
    "The initial condition layer"
    initial_conditions::U0
    "The control parameter collection"
    controls::C
end

function SingleShootingLayer(initial_conditions::InitialCondition, controls::ControlParameters; algorithm::SciMLBase.AbstractDEAlgorithm, name = gensym(:single_shooting))
    return SingleShootingLayer{typeof(algorithm), typeof(initial_conditions), typeof(controls)}(name, algorithm, initial_conditions, controls)
end

function SingleShootingLayer(initial_conditions::InitialCondition, controls...; algorithm::SciMLBase.AbstractDEAlgorithm, name = gensym(:single_shooting))
    controls = ControlParameters(controls...)
    return SingleShootingLayer{typeof(algorithm), typeof(initial_conditions), typeof(controls)}(name, algorithm, initial_conditions, controls)
end

function SingleShootingLayer(problem::SciMLBase.DEProblem, controls...; algorithm::SciMLBase.AbstractDEAlgorithm, name = gensym(:single_shooting), kwargs...)
    _, repack, _ = SciMLStructures.canonicalize(
        SciMLStructures.Tunable(), problem.p
    )
    initial_conditions = InitialCondition(problem; kwargs...)
    controls = ControlParameters(controls..., transform = (nt) -> repack(reduce(vcat, map(collect, nt))))
    return SingleShootingLayer{typeof(algorithm), typeof(initial_conditions), typeof(controls)}(name, algorithm, initial_conditions, controls)
end

_subscript(i::Integer) = (i |> digits |> reverse .|> dgt -> Char(0x2080 + dgt)) |> join

function default_system(problem::SciMLBase.DEProblem, controls)
    states = [Symbol(:x, _subscript(i)) for i in eachindex(problem.u0)]
    ps = collect(keys(controls.controls))
    t = :t
    return SymbolCache(states, ps, t)
end

function get_new_system(problem, controls)
    sys = something(
        symbolic_container(problem.f),
        default_system(problem, controls)
    )
    return remake_system(sys, controls)
end

function remake_system(sys::SymbolCache, controls)
    return SymbolCache(
        variable_symbols(sys), parameter_symbols(sys), independent_variable_symbols(sys);
        timeseries_parameters = Dict(
            [c.name => ParameterTimeseriesIndex(i, 1) for (i, c) in enumerate(values(controls.controls))]
        )
    )
end

function LuxCore.initialstates(rng::Random.AbstractRNG, layer::SingleShootingLayer)
    (; initial_conditions, controls) = layer
    timegrid = vcat(Corleone.get_timegrid(initial_conditions), Corleone.get_timegrid(controls))
    t0, tinf = get_tspan(initial_conditions)
    timegrid = filter(t -> t >= t0 && t <= tinf, timegrid)
    unique!(sort!(timegrid))
    timegrid = collect(zip(timegrid[1:(end - 1)], timegrid[2:end]))
    # We bin the timegrid now to avoid recursion errors
    N = length(timegrid)
    partions = vcat(collect(1:MAXBINSIZE:N), N)
    unique!(partions)
    timegrid = ntuple(i -> Tuple(timegrid[partions[i]:partions[i + 1]]), length(partions) - 1)
    # Define the system for the symbolic indexing interface
    sys = get_new_system(initial_conditions.problem, controls)
    return (;
        timestops = timegrid,
        initial_conditions = LuxCore.initialstates(rng, initial_conditions),
        controls = LuxCore.initialstates(rng, controls),
        system = sys,
    )
end

function (layer::SingleShootingLayer)(::Any, ps, st)
    (; algorithm, initial_conditions, controls) = layer
    problem, st_ic = initial_conditions(nothing, ps.initial_conditions, st.initial_conditions)
    inputs, st_controls = controls(st.timestops, ps.controls, st.controls)
    solutions = eval_problem(problem, algorithm, true, inputs)
    return Trajectory(layer, solutions, merge(st, (; controls = st_controls)))
end

@generated function _eval_problem(problem, algorithm, save_start, trajectory::Tuple{Vararg{NamedTuple, N}}) where {N}
    sols = [gensym() for _ in Base.OneTo(N)]
    exprs = Expr[]
    for i in Base.OneTo(N)
        push!(
            exprs, :(
                sol = solve(
                    problem, algorithm,
                    p = trajectory[$(i)].p,
                    tspan = trajectory[$i].tspan, save_everystep = false, save_start = $(i == 1) && save_start, save_end = true
                )
            )
        )
        push!(exprs, :(problem = remake(problem, u0 = sol.u[end])))
        push!(exprs, :($(sols[i]) = (; p = trajectory[$(i)].p, u = sol.u, t = sol.t)))
    end
    push!(exprs, :(return ($(sols...),), problem))
    ex = Expr(:block, exprs...)
    return ex
end

function eval_problem(problem, algorithm, save_start, trajectory::Tuple)
    current_solution, problem = _eval_problem(problem, algorithm, save_start, Base.first(trajectory))
    length(trajectory) == 1 && return current_solution
    return (
        current_solution...,
        eval_problem(
            problem,
            algorithm, false, Base.tail(trajectory)
        )...,
    )
end


function Trajectory(::SingleShootingLayer, solutions, st)
    (; system) = st
    u = reduce(vcat, map(sol -> sol.u, solutions))
    t = reduce(vcat, map(sol -> sol.t, solutions))
    p = reduce(vcat, map(sol -> sol.p, solutions))
    t_controls = reduce(vcat, map(sol -> first(sol.t), solutions))
    controlseries = ParameterTimeseriesCollection((ControlSignal(t_controls, p),), deepcopy(first(p)))
    p = deepcopy(first(p))
    return Trajectory{typeof(system), typeof(u), typeof(p), typeof(t), typeof(controlseries), Nothing}(system, u, p, t, controlseries, nothing), st
end
