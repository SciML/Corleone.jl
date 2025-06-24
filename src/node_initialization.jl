function __fallbackdefault(x)
    if ModelingToolkit.hasbounds(x)
        lo, hi = ModelingToolkit.getbounds(x)
        return (hi - lo) / 2
    else Symbolics.hasmetadata(x, Symbolics.VariableDefaultValue)
        return Symbolics.getdefaultval(x)
    end
    return zero(Symbolics.symtype(x))
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

function (f::RandomInitialization)(problem::SciMLBase.AbstractSciMLProblem, args...; kwargs...)
    (; rng) = f
    psyms = SciMLBase.getparamsyms(problem)
    shooting_vars = filter(is_shootingvariable, psyms)
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

function (f::ForwardSolveInitialization)(problem::SciMLBase.AbstractSciMLProblem, alg, args...; kwargs...)
    psyms = SciMLBase.getparamsyms(problem)
    shooting_vars = filter(is_shootingvariable, psyms)
    isempty(shooting_vars) && return problem
    shooting_points = filter(is_shootingpoint, psyms)
    timepoints = unique!(sort!(reduce(vcat, Symbolics.getdefaultval.(shooting_points))))
    tspan = problem.tspan
    tstops = get(problem.kwargs, :tstpops, collect(tspan))
    new_tspan = (min(minimum(tstops) - eps(), first(tspan)), last(tspan)) # Some hacky trick to take all callbacks into account
    prob = remake(problem, tspan=new_tspan)
    sol = solve(prob, alg; kwargs...)
    foreach(shooting_vars) do si
        xi = get_shootingparent(si)
        if is_statevar(xi)
            newvars = sol(timepoints)[xi]
            SymbolicIndexingInterface.setp(problem, si)(problem, newvars)
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

function (f::FunctionInitialization)(problem::SciMLBase.AbstractSciMLProblem, args...; kwargs...)
    psyms = SciMLBase.getparamsyms(problem)
    shooting_vars = filter(is_shootingvariable, psyms)
    isempty(shooting_vars) && return problem
    shooting_points = filter(is_shootingpoint, psyms)
    timepoints = unique!(sort!(reduce(vcat, Symbolics.getdefaultval.(shooting_points))))
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

function linear_initializer(u_inf, problem, t)
    (_, tinf) = problem.tspan
    u0 = problem.u0
    slope = u_inf .- u0
    val = t ./ tinf
    u0 .+ slope .* val
end

"""
$(FUNCTIONNAME)

Creates a (`FunctionInitialization`)[@ref] with linearly interpolates between u0 and the provided u_inf.
"""
function LinearInterpolationInitialization(u0_inf)
    finit = Base.Fix1(linear_initializer, u0_inf)
    FunctionInitialization{typeof(finit)}(finit)
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

function (f::CustomInitialization)(problem::SciMLBase.AbstractSciMLProblem, args...; kwargs...)
    psyms = SciMLBase.getparamsyms(problem)
    shooting_vars = filter(is_shootingvariable, psyms)
    isempty(shooting_vars) && return problem
    shooting_points = filter(is_shootingpoint, psyms)
    timepoints = unique!(sort!(reduce(vcat, Symbolics.getdefaultval.(shooting_points))))
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

function (f::ConstantInitialization)(problem::SciMLBase.AbstractSciMLProblem, args...; kwargs...)
    psyms = SciMLBase.getparamsyms(problem)
    shooting_vars = filter(is_shootingvariable, psyms)
    isempty(shooting_vars) && return problem
    shooting_points = filter(is_shootingpoint, psyms)
    timepoints = unique!(sort!(reduce(vcat, Symbolics.getdefaultval.(shooting_points))))
    (; initial_values) = f
    foreach(shooting_vars) do si
        xi = get_shootingparent(si)
        if is_statevar(xi)
            newvar = get(initial_values, xi) do
                __fallbackdefault(xi)
            end
            SymbolicIndexingInterface.setp(problem, si)(problem, collect(fill(newvar, length(timepoints))))
        end
    end
    return problem
end