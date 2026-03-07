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

function SingleShootingLayer(initial_conditions::InitialCondition, controls::ControlParameters; algorithm::SciMLBase.AbstractDEAlgorithm , name=gensym(:single_shooting))
    return SingleShootingLayer{typeof(algorithm), typeof(initial_conditions), typeof(controls)}(name, algorithm, initial_conditions, controls)
end

function SingleShootingLayer(initial_conditions::InitialCondition, controls...; algorithm::SciMLBase.AbstractDEAlgorithm , name=gensym(:single_shooting))
    controls = ControlParameters(controls...)
    return SingleShootingLayer{typeof(algorithm), typeof(initial_conditions), typeof(controls)}(name, algorithm, initial_conditions, controls)
end

function SingleShootingLayer(problem::SciMLBase.DEProblem, controls...; algorithm::SciMLBase.AbstractDEAlgorithm , name=gensym(:single_shooting), kwargs...)
    _, repack, _ = SciMLStructures.canonicalize(
        SciMLStructures.Tunable(), problem.p
    )
    initial_conditions = InitialCondition(problem; kwargs...)
    controls = ControlParameters(controls..., transform = (nt) -> repack(collect(values(nt))))
    return SingleShootingLayer{typeof(algorithm), typeof(initial_conditions), typeof(controls)}(name, algorithm, initial_conditions, controls)
end

_subscript(i::Integer) = (i |> digits |> reverse .|> dgt -> Char(0x2080 + dgt)) |> join

function default_system(problem::SciMLBase.DEProblem, controls)
    states = [Symbol(:x, _subscript(i)) for i in eachindex(problem.u0)]
    ps = collect(keys(controls.controls))
    t = :t 
    SymbolCache(states, ps, t)
end

function get_new_system(problem, controls)
    sys = something(
        symbolic_container(problem.f), 
        default_system(problem, controls)
    )
    remake_system(sys, controls)
end

function remake_system(sys::SymbolCache, controls)
    SymbolCache(variable_symbols(sys), parameter_symbols(sys), independent_variable_symbols(sys); 
        timeseries_parameters = Dict(
            [c.name => ParameterTimeseriesIndex(i, 1) for (i,c) in enumerate(values(controls.controls))]
        )
    )
end

function LuxCore.initialstates(rng::Random.AbstractRNG, layer::SingleShootingLayer)
    (; initial_conditions, controls) = layer
    timegrid = vcat(Corleone.get_timegrid(initial_conditions), Corleone.get_timegrid(controls)) 
    t0, tinf = get_tspan(initial_conditions)
    timegrid = filter(t -> t >= t0 && t <= tinf, timegrid)
    unique!(sort!(timegrid))
    # We bin the timegrid now to avoid recursion errors 
    N = length(timegrid)
    partions = vcat(collect(1:MAXBINSIZE:N), N)
    unique!(partions)
    timegrid = ntuple(i->Tuple(timegrid[partions[i]:partions[i+1]]), length(partions)-1)
    # Define the system for the symbolic indexing interface 
    sys = get_new_system(initial_conditions.problem, controls)
    return (; 
        timestops = timegrid, 
        initial_conditions = LuxCore.initialstates(rng, initial_conditions),
        controls = LuxCore.initialstates(rng, controls),
        system = sys, 
        control_getters = getsym(sys, keys(controls.controls)), 
        state_getters = getsym(sys, variable_symbols(sys)),        
    )
end

function (layer::SingleShootingLayer)(::Any, ps, st)
    (; algorithm, initial_conditions, controls) = layer
    # We evaluate the problem
    problem, st_ic = initial_conditions(nothing, ps.initial_conditions, st.initial_conditions)
    solutions = eval_problem(problem, algorithm, controls, ps.controls, st.controls, true, st.timestops...)
    return Trajectory(layer, solutions, ps,  merge(st, (; initial_conditions = st_ic,)))
end


@generated function eval_problem(problem, algorithm, controls, ps, st, save_start, timestops::NTuple{N,<:Real}) where N 
    sols  = [gensym() for _ in Base.OneTo(N-1)]
    sts = [gensym() for _ in Base.OneTo(N-1)]
    psym = gensym()
    exprs = Expr[] 
    for i in Base.OneTo(N-1)
         push!(exprs, :(($(psym), $(sts[i])) = controls(timestops[$i], ps, st))) 
         push!(exprs, :($(sols[i]) = solve(problem, algorithm, p=$psym, tspan = (timestops[$i], timestops[$(i+1)]), save_everystep=false, save_start= $(i == 1) && save_start, save_end=true))) 
         push!(exprs, :(problem = remake(problem, u0=last($(sols[i])))))
    end
    
    push!(exprs, Expr(:tuple, sols...))
    ex = Expr(:block, exprs...)
    return ex
end

function eval_problem(problem, algorithm, controls, ps, st, save_start, current, timestops...)
    current = eval_problem(problem, algorithm, controls, ps, st, save_start, current)
    (current..., eval_problem(remake(problem ,u0 = last(last(current))), algorithm, controls, ps, st, false, timestops...)...)
end

function eval_problem(problem, algorithm, controls, ps, st, save_start, timestops)
    current = eval_problem(problem, algorithm, controls, ps, st, save_start, timestops)
end

function _flatten_states(solutions::NTuple{N,S}) where {N,S}
    U = eltype(first(solutions).u)
    total = sum(sol -> length(sol.u), solutions)
    out = Vector{U}(undef, total)
    k = firstindex(out)
    for sol in solutions
        for ui in sol.u
            out[k] = ui
            k += 1
        end
    end
    return out
end

function _flatten_times(solutions::NTuple{N,S}) where {N,S}
    Tt = eltype(first(solutions).t)
    total = sum(sol -> length(sol.t), solutions)
    out = Vector{Tt}(undef, total)
    k = firstindex(out)
    for sol in solutions
        for ti in sol.t
            out[k] = ti
            k += 1
        end
    end
    return out
end

function _eval_control_signal(control::ControlParameter, p, st, t)
    values = Vector{eltype(p)}(undef, length(t))
    st_local = st
    for i in eachindex(t)
        values[i], st_local = control(t[i], p, st_local)
    end
    return values
end

@generated function _control_timeseries(controls::NamedTuple{names}, ps::NamedTuple{names}, st::NamedTuple{names}, t) where {names}
    exprs = map(names) do nm
        :(ControlSignal(t, _eval_control_signal(controls.$nm, ps.$nm, st.$nm, t)))
    end
    return Expr(:tuple, exprs...)
end

function Trajectory(layer::SingleShootingLayer, solutions, ps, st)
    (; system, ) = st
    controls = layer.controls.controls
    u = _flatten_states(solutions)
    t = _flatten_times(solutions)
    cseries = _control_timeseries(controls, ps.controls, st.controls, t)
    p = first(solutions).prob.p
    controlseries = ParameterTimeseriesCollection(cseries, deepcopy(p))
    Trajectory{typeof(system), typeof(u), typeof(p), typeof(t), typeof(controlseries), Nothing}(system, u, p, t, controlseries, nothing)
end