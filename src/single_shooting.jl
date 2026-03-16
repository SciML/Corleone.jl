"""
$(TYPEDEF)

Single-shooting layer coupling initial conditions and controls for trajectory simulation.

# Fields
$(FIELDS)
"""
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

function SciMLBase.remake(layer::SingleShootingLayer; kwargs...)
    initial_conditions = get(kwargs, :initial_conditions, remake(layer.initial_conditions; kwargs...))
    controls = get(kwargs, :controls, remake(layer.controls; kwargs...))
    algorithm = get(kwargs, :algorihtm, layer.algorithm)
    name = get(kwargs, :name, layer.name)
    return SingleShootingLayer{
        typeof(algorithm),
        typeof(initial_conditions), typeof(controls),
    }(
        name, algorithm, initial_conditions, controls
    )
end

"""
$(SIGNATURES)

Construct a [`SingleShootingLayer`](@ref) from pre-built initial-condition and control layers.
"""
function SingleShootingLayer(initial_conditions::InitialCondition, controls::ControlParameters; algorithm::SciMLBase.AbstractDEAlgorithm, name = gensym(:single_shooting))
    return SingleShootingLayer{typeof(algorithm), typeof(initial_conditions), typeof(controls)}(name, algorithm, initial_conditions, controls)
end

"""
$(SIGNATURES)

Construct a [`SingleShootingLayer`](@ref) from an initial-condition layer and control specifications.
"""
function SingleShootingLayer(initial_conditions::InitialCondition, controls...; algorithm::SciMLBase.AbstractDEAlgorithm, name = gensym(:single_shooting))
    controls = ControlParameters(controls...)
    return SingleShootingLayer{typeof(algorithm), typeof(initial_conditions), typeof(controls)}(name, algorithm, initial_conditions, controls)
end

"""
$(SIGNATURES)

Construct a [`SingleShootingLayer`](@ref) directly from a `DEProblem` and control specifications.
"""
function SingleShootingLayer(problem::SciMLBase.DEProblem, controls...; algorithm::SciMLBase.AbstractDEAlgorithm, name = gensym(:single_shooting), kwargs...)

    repack = let p0 = problem.p
        (x) -> SciMLStructures.replace(SciMLStructures.Tunable(), p0, vcat(values(x)...))
    end
    initial_conditions = InitialCondition(problem; kwargs...)
    controls = ControlParameters(controls..., transform = repack)
    return SingleShootingLayer{typeof(algorithm), typeof(initial_conditions), typeof(controls)}(name, algorithm, initial_conditions, controls)
end


"""
$(SIGNATURES)

Return unicode subscript string for positive integer `i`.
"""
_subscript(i::Integer) = (i |> digits |> reverse .|> dgt -> Char(0x2080 + dgt)) |> join

"""
$(SIGNATURES)

Build a default symbolic system cache when the problem has no symbolic container.
"""
function default_system(problem::SciMLBase.DEProblem, controls)
    states = [Symbol(:x, _subscript(i)) for i in eachindex(problem.u0)]
    ps = collect(keys(controls.controls))
    t = :t
    return SymbolCache(states, ps, t)
end

"""
$(SIGNATURES)

Return a symbolic system with control parameters registered as time series.
"""
function get_new_system(problem, controls)
    sys = symbolic_container(problem.f)
    if isnothing(sys) || isnothing(variable_symbols(sys)) || isnothing(parameter_symbols(sys))
        sys = default_system(problem, controls)
    end
    try
        return remake_system(sys, controls)
    catch
        return remake_system(default_system(problem, controls), controls)
    end
end

"""
$(SIGNATURES)

Rebuild `sys` with control names mapped to `ParameterTimeseriesIndex` entries.
"""
function remake_system(sys::SymbolCache, controls)
    return SymbolCache(
        variable_symbols(sys), parameter_symbols(sys), independent_variable_symbols(sys);
        timeseries_parameters = Dict(
            [c.name => ParameterTimeseriesIndex(1, i) for (i, c) in enumerate(values(controls.controls))]
        )
    )
end

get_problem(layer::SingleShootingLayer) = get_problem(layer.initial_conditions)
get_tspan(layer::SingleShootingLayer) = get_tspan(layer.initial_conditions)
get_quadrature_indices(layer::SingleShootingLayer) = get_quadrature_indices(layer.initial_conditions)
get_tunable_u0(layer::SingleShootingLayer, full::Bool = false) = get_tunable_u0(layer.initial_conditions, full)

function get_shooting_variables(layer::SingleShootingLayer)
    problem = get_problem(layer)
    tunable_ic = get_tunable_u0(layer)
    cnames = [c.name for c in layer.controls.controls if is_shooted(c)]
    return (; state = tunable_ic, control = cnames)
end

function get_timegrid(layer::SingleShootingLayer)
    (; initial_conditions, controls) = layer
    timegrid = vcat(Corleone.get_timegrid(initial_conditions), Corleone.get_timegrid(controls))
    t0, tinf = get_tspan(initial_conditions)
    timegrid = filter(t -> t >= t0 && t <= tinf, timegrid)
    unique!(sort!(timegrid))
    return timegrid
end

"""
$(SIGNATURES)

Initialize runtime state for single-shooting evaluation, including binned time stops.
"""
function LuxCore.initialstates(rng::Random.AbstractRNG, layer::SingleShootingLayer)
    (; initial_conditions, controls) = layer
    t0, tinf = get_tspan(initial_conditions)
    timegrid = get_timegrid(layer)
    timegrid = collect(zip(timegrid[1:(end - 1)], timegrid[2:end]))
    # We bin the timegrid now to avoid recursion errors
    N = length(timegrid)
    if N == 0
        timegrid = [(t0, tinf)]
        N = 1
    end
    partitions = collect(1:MAXBINSIZE:N)
    if isempty(partitions) || last(partitions) != (N + 1)
        push!(partitions, N + 1)
    end
    timegrid = ntuple(i -> Tuple(timegrid[partitions[i]:(partitions[i + 1] - 1)]), length(partitions) - 1)
    # Define the system for the symbolic indexing interface
    sys = get_new_system(initial_conditions.problem, controls)
    return (;
        timestops = timegrid,
        initial_conditions = LuxCore.initialstates(rng, initial_conditions),
        controls = LuxCore.initialstates(rng, controls),
        system = sys,
    )
end

"""
$(SIGNATURES)

Evaluate the layer and return a [`Trajectory`](@ref).
"""
function (layer::SingleShootingLayer)(::Any, ps, st)
    (; algorithm, initial_conditions, controls) = layer
    problem, st_ic = initial_conditions(nothing, ps.initial_conditions, st.initial_conditions)
    inputs, st_controls = controls(st.timestops, ps.controls, st.controls)
    solutions = eval_problem(problem, algorithm, true, inputs)
    return Trajectory(layer, solutions, merge(st, (; controls = st_controls)))
end

"""
$(SIGNATURES)

Generated helper that solves one tuple block of trajectory intervals.
"""
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
        push!(exprs, :($(sols[i]) = (; p = trajectory[$(i)].p, u = sol.u, t = sol.t, tspan = trajectory[$i].tspan)))
    end
    push!(exprs, :(return ($(sols...),), problem))
    ex = Expr(:block, exprs...)
    return ex
end

"""
$(SIGNATURES)

Recursively evaluate all trajectory bins for a single-shooting problem.
"""
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


"""
$(SIGNATURES)

Construct a [`Trajectory`](@ref) from solved single-shooting segments.
"""
function Trajectory(::SingleShootingLayer, solutions, st)
    (; system) = st
    u = _collect(solutions, :u) #reduce(vcat, map(sol -> sol.u, solutions), init = eltype(first(solutions).u)[])
    t = _collect(solutions, :t) #reduce(vcat, map(sol -> sol.t, solutions), init = eltype(first(solutions).t)[])
    t_controls = _collect(solutions, :tspan, first)
    p = collect(map(sol -> sol.p, solutions))
    controlseries = ParameterTimeseriesCollection((ControlSignal(t_controls, p),), deepcopy(first(p)))
    p = deepcopy(first(p))
    return Trajectory{typeof(system), typeof(u), typeof(p), typeof(t), typeof(controlseries), Nothing}(system, u, p, t, controlseries, nothing), st
end

function _collect(solutions, sym::Symbol, f::Function = identity)
    xs = map(f ∘ Base.Fix2(getproperty, sym), solutions)
    return vcat(xs...)
end
