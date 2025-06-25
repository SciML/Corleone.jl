function __fallbackdefault(x)
    if ModelingToolkit.hasbounds(x)
        lo, hi = ModelingToolkit.getbounds(x)
        return (hi - lo) / 2
    else Symbolics.hasmetadata(x, Symbolics.VariableDefaultValue)
        return Symbolics.getdefaultval(x)
    end
    return zero(Symbolics.symtype(x))
end

function __default_shooting_vars(prob::SciMLBase.AbstractSciMLProblem)
    psyms = SciMLBase.getparamsyms(prob)
    return filter(is_shootingvariable, psyms)
end

function __shooting_timepoints(prob::SciMLBase.AbstractSciMLProblem)
    psyms = SciMLBase.getparamsyms(prob)
    shooting_points = filter(is_shootingpoint, psyms)
    unique!(sort!(reduce(vcat, Symbolics.getdefaultval.(shooting_points))))
end

"""
$(TYPEDEF)

Abstract type defining different formulations for initialization of shooting node variables.

"""
abstract type AbstractNodeInitialization end

function (f::AbstractNodeInitialization)(problem::SciMLBase.AbstractSciMLProblem, args...; kwargs...)
    throw(ArgumentError("The initialization $f is not implemented."))
end

struct DefaultsInitialization <: AbstractNodeInitialization end
(::DefaultsInitialization)(problem::SciMLBase.AbstractSciMLProblem, args...; kwargs...) = problem

"""
$(TYPEDEF)

Initializes the problem with random values in the bounds of the variables.

# Fields
$(FIELDS)
"""
struct RandomInitialization{R<:Random.AbstractRNG} <: AbstractNodeInitialization
    "The random number generator"
    rng::R
end

RandomInitialization() = RandomInitialization(Random.default_rng())

function (f::RandomInitialization)(problem::SciMLBase.AbstractSciMLProblem, args...;
            shooting_vars = __default_shooting_vars(problem), kwargs...)
    (; rng) = f
    isempty(shooting_vars) && return problem
    foreach(shooting_vars) do si
        xi = get_shootingparent(si)
        if (is_statevar(xi) && hasbounds(xi))
            lo, hi = ModelingToolkit.getbounds(xi)
            newvars = lo .+ rand(rng, typeof(hi), size(si)) .* hi
            SymbolicIndexingInterface.setp(problem, si)(problem, newvars)
        end
    end
    problem
end

"""
$(TYPEDEF)

Initializes the problem using a single forward solve of the problem.
"""
struct ForwardSolveInitialization <: AbstractNodeInitialization end

function (f::ForwardSolveInitialization)(problem::SciMLBase.AbstractSciMLProblem, alg, args...;
            shooting_vars = __default_shooting_vars(problem), kwargs...)

    default_shooting_vars = __default_shooting_vars(problem)

    filter_vars = map(x -> x in shooting_vars, default_shooting_vars)

    dependent_shooting_vars =  default_shooting_vars[filter_vars]
    independent_shooting_vars = default_shooting_vars[.!filter_vars]

    isempty(shooting_vars) && return problem
    timepoints = __shooting_timepoints(problem)

    independent_defaults = map(independent_shooting_vars) do var
        SymbolicIndexingInterface.getp(problem, var)(problem)
    end

    local_u0 = copy(problem.u0)
    newvars = reduce(hcat, map(enumerate(zip(timepoints[1:end-1], timepoints[2:end]))) do (i,local_tspan)
        for (var_idx,indep_var) in enumerate(independent_shooting_vars)
            idx_indep = SymbolicIndexingInterface.variable_index(problem, get_shootingparent(indep_var))
            local_u0[idx_indep] = independent_defaults[var_idx][i]
        end
        _prob = remake(problem, u0=local_u0, tspan= local_tspan)
        _sol = solve(_prob, alg; kwargs...)
        local_u0 = last(_sol)
        reduce(vcat, map(dependent_shooting_vars) do var
            idx_dep = SymbolicIndexingInterface.variable_index(problem, get_shootingparent(var))
            local_u0[idx_dep]
        end)
    end)


    foreach(enumerate(dependent_shooting_vars)) do (i,si)
        xi = get_shootingparent(si)
        if is_statevar(xi)
            _newvars = vcat(__fallbackdefault(xi), newvars[i,:])
            SymbolicIndexingInterface.setp(problem, si)(problem, _newvars)
        end
    end
    problem
end

"""
$(TYPEDEF)

Initializes the problem using a custom function which returns a vector for all variables.

# Fields
$(FIELDS)
"""
struct FunctionInitialization{F} <: AbstractNodeInitialization
    "Initialization function f(problem, t_i) -> u(t_i)"
    initializer::F
end

function (f::FunctionInitialization)(problem::SciMLBase.AbstractSciMLProblem, args...;
            shooting_vars=__default_shooting_vars(problem), kwargs...)
    isempty(shooting_vars) && return problem
    timepoints = __shooting_timepoints(problem)
    (; initializer) = f
    u0s = reduce(hcat, map(timepoints) do ti
        initializer(problem, ti)
    end)
    sysvars = SciMLBase.getsyms(problem)
    foreach(shooting_vars) do si
        xi = get_shootingparent(si)
        if is_statevar(xi)
            id = findfirst(Base.Fix1(isequal, xi), sysvars)
            newvars = u0s[id, :]
            SymbolicIndexingInterface.setp(problem, si)(problem, newvars)
        end
    end
end

function linear_initializer(u0, u_inf, t, tspan)
    t0, t_inf = tspan
    slope = u_inf .- u0
    val = (t-t0) ./ t_inf
    u0 .+ slope .* val
end

"""
$(TYPEDEF)

Initializes the problem using a custom function which returns a vector for all variables.

# Fields
$(FIELDS)
"""
struct LinearInterpolationInitialization{T <: AbstractDict} <: AbstractNodeInitialization
    "Terminal values for linear interpolation of initial and terminal values."
    terminal_values::T
end

"""
$(FUNCTIONNAME)

Creates a (`FunctionInitialization`)[@ref] with linearly interpolates between u0 and the provided u_inf.
"""
function (f::LinearInterpolationInitialization)(problem::SciMLBase.AbstractSciMLProblem, args...;
            shooting_vars=__default_shooting_vars(problem), kwargs...)
    isempty(shooting_vars) && return problem
    timepoints = __shooting_timepoints(problem)
    foreach(shooting_vars) do si
        xi = get_shootingparent(si)
        if is_statevar(xi)
            u0 = __fallbackdefault(xi)
            uinf = f.terminal_values[xi]
            newvar = map(timepoints) do ti
                linear_initializer(u0, uinf, ti, problem.tspan)
            end
            SymbolicIndexingInterface.setp(problem, si)(problem, newvar)
        end
    end
    return problem
end

"""
$(TYPEDEF)

Initializes the system with a custom vector of points provided as a Dictionary of variable => values.

# Fields
$(FIELDS)

# Note
If the variable is not present in the dictionary, we use the fallback value.
"""
struct CustomInitialization{I <: AbstractDict} <: AbstractNodeInitialization
    "The init values for all dependent variables"
    initial_values::I
end

function (f::CustomInitialization)(problem::SciMLBase.AbstractSciMLProblem, args...;
            shooting_vars=__default_shooting_vars(problem), kwargs...)
    isempty(shooting_vars) && return problem
    timepoints = __shooting_timepoints(problem)
    (; initial_values) = f
    foreach(shooting_vars) do si
        xi = get_shootingparent(si)
        if is_statevar(xi)
            newvar = get(initial_values, xi) do
                fill(__fallbackdefault(xi), length(timepoints)) |> collect
            end
            SymbolicIndexingInterface.setp(problem, si)(problem, newvar)
        end
    end
    return problem
end

"""
$(TYPEDEF)

Initializes the system with a custom single value provided as a Dictionary of variable => values.

# Fields
$(FIELDS)

# Note
If the variable is not present in the dictionary, we use the fallback value.
"""
struct ConstantInitialization{I <: AbstractDict} <: AbstractNodeInitialization
    "The init values for all dependent variables"
    initial_values::I
end

function (f::ConstantInitialization)(problem::SciMLBase.AbstractSciMLProblem,  args...;
            shooting_vars = __default_shooting_vars(problem), kwargs...)
    isempty(shooting_vars) && return problem
    timepoints = __shooting_timepoints(problem)
    (; initial_values) = f
    foreach(shooting_vars) do si
        xi = get_shootingparent(si)
        if is_statevar(xi)

            newvar = get(initial_values, xi) do
                __fallbackdefault(xi)
            end
            newinit = vcat(__fallbackdefault(xi), collect(fill(newvar, length(timepoints)-1)))
            SymbolicIndexingInterface.setp(problem, si)(problem, newinit)
        end
    end
    return problem
end

struct HybridInitialization{P <: Dict} <: AbstractNodeInitialization
    "Pair of variables and corresponding AbstractNodeInitialization methods"
    inits::P
    "Init method for remaining variables"
    default_init::AbstractNodeInitialization
end


function (f::AbstractNodeInitialization)(predictor::OCPredictor; kwargs...)
    newprob = f(predictor.problem, predictor.alg; kwargs...)
    @set predictor.problem = newprob
end



function (f::HybridInitialization)(predictor::OCPredictor; kwargs...)
    (; problem, alg) = predictor

    psyms = SciMLBase.getparamsyms(problem)
    shooting_vars = filter(is_shootingvariable, psyms)

    defined_vars = [x.first for x in f.inits]

    forward_involved = [typeof(x.second) <: ForwardSolveInitialization for x in f.inits]
    forward_default = typeof(f.default_init) <: ForwardSolveInitialization

    any_forward = any(forward_involved) || forward_default

    forward_vars = any_forward ? reduce(vcat, defined_vars[forward_involved]) : []
    defined_vars = reduce(vcat, defined_vars)

    rest = map(x -> all(!Base.Fix1(isequal, x), defined_vars), get_shootingparent.(shooting_vars))

    remaining_vars = shooting_vars[rest]

    forward_vars = forward_default ? vcat(forward_vars. get_shootingparent.(remaining_vars)) : forward_vars
    corresponding_fwd_vars = map(forward_vars) do v
        shooting_vars[findfirst([isequal(get_shootingparent(x),v) for x in shooting_vars])]
    end

    init_copy = copy(f.inits)
    init_copy = any_forward ? delete!(init_copy, ForwardSolveInitialization()) : init_copy

    for p in init_copy
        corresponding_shooting = map(p.first) do var
            shooting_vars[findfirst([isequal(get_shootingparent(x),var) for x in shooting_vars])]
        end
        newprob = p.second(predictor.problem, alg; shooting_vars=corresponding_shooting, kwargs...)
        @set predictor.problem = newprob
    end
    newprob = any_forward ? ForwardSolveInitialization()(predictor.problem, alg;
                shooting_vars = corresponding_fwd_vars, kwargs...) : predictor.problem

    @set predictor.problem = newprob
end
