"""
$(TYPEDEF)

Abstract type defining different formulations for initialization of shooting node variables.

"""
abstract type AbstractNodeInitialization end


struct ForwardSolveInitialization{I,S} <: AbstractNodeInitialization
    "The init values for all dependent variables"
    init::I
    solver_info::S
end

function ForwardSolveInitialization(sys, timepoints, alg; init_values::Union{AbstractVector{<:Pair}, Nothing}=nothing, kwargs...)
    vars = unknowns(sys)
    idx =  .!is_costvariable.(vars) .&& .!isinput.(vars)
    init_vars = vars[idx]

    prob = ODEProblem(complete(sys); allow_cost = true)
    tstops = [last(de.condition.arguments) |> Symbolics.getdefaultval for de in discrete_events(sys)]
    tspan = ModelingToolkit.get_tspan(sys)
    new_tspan = (min(minimum(tstops)-eps(), first(tspan)), last(tspan)) # Some hacky trick to take all callbacks into account
    prob = isempty(tstops) ? prob : remake(prob, tspan=new_tspan)
    sol = solve(prob, alg; tstops=tstops, kwargs...)
    _sol = Array(sol(timepoints, idxs=findall(idx)))
    init = init_vars .=> eachrow(_sol)
    solver_info = (alg = alg, solver_kwargs = kwargs)
    return ForwardSolveInitialization{typeof(init), typeof(solver_info)}(init, solver_info)
end


struct DefaultsInitialization{I} <: AbstractNodeInitialization
    "The init values for all dependent variables"
    init::I
end

function DefaultsInitialization(sys, timepoints;  init_values::Union{AbstractVector{<:Pair}, Nothing}=nothing)
    vars = unknowns(sys)
    idx =  .!is_costvariable.(vars) .&& .!isinput.(vars)
    init_vars = vars[idx]

    init = map(init_vars) do x
                u0 = Symbolics.hasmetadata(x, Symbolics.VariableDefaultValue) ? Symbolics.getdefaultval(x) : 0.0
                x => [u0 for _ in 1:length(timepoints)]
    end
    return DefaultsInitialization{typeof(init)}(init)
end

struct RandomInitialization{I} <: AbstractNodeInitialization
    "The init values for all dependent variables"
    init::I
end

function RandomInitialization(sys, timepoints;  init_values::Union{AbstractVector{<:Pair}, Nothing}=nothing)
    vars = unknowns(sys)
    idx =  .!is_costvariable.(vars) .&& .!isinput.(vars)
    init_vars = vars[idx]
    bounds = getbounds.(init_vars)
    default_values = Symbolics.getdefaultval.(init_vars)
    init = [var => vcat(default_values[i], rand(LinRange(bounds[i]..., 100), length(timepoints)-1)) for (i,var) in enumerate(init_vars)]
    return RandomInitialization{typeof(init)}(init)
end

struct LinearInterpolationInitialization{I} <: AbstractNodeInitialization
    "The init values for all dependent variables"
    init::I
end

function LinearInterpolationInitialization(sys, timepoints;  init_values::Union{AbstractVector{<:Pair}, Nothing}=nothing)
    vars = unknowns(sys)
    idx =  .!is_costvariable.(vars) .&& .!isinput.(vars)
    init_vars = vars[idx]

    @assert all([length(x.second) == 1 for x in init_values]) "Initialization via linear interpolation
        expected one value for the state at the end point, got $([length(x.second) for x in init_values]) values."
    init = map(init_vars) do var
        default_value = Symbolics.getdefaultval.(var)
        slope = 0.0
        for _pair in init_values
            isequal(var, _pair.first) || continue
            slope = only(_pair.second) - default_value
        end
        var => default_value .+ slope * (timepoints ./ last(timepoints))
    end
    return LinearInterpolationInitialization{typeof(init)}(init)
end


struct CustomInitialization{I} <: AbstractNodeInitialization
    "The init values for all dependent variables"
    init::I
end

function CustomInitialization(sys, timepoints;  init_values::Union{AbstractVector{<:Pair}, Nothing}=nothing)
    @assert all([length(x.second) == length(timepoints) for x in init_values]) "Custom initialization
            expected $(length(timepoints)) values, got $([length(x.second) for x in init_values]) values."
    CustomInitialization{typeof(init_values)}(init_values)
end

struct ConstantInitialization{I} <: AbstractNodeInitialization
    "The init values for all dependent variables"
    init::I
end

function ConstantInitialization(sys, timepoints; init_values::Union{AbstractVector{<:Pair}, Nothing}=nothing)
    vars = unknowns(sys)
    idx =  .!is_costvariable.(vars) .&& .!isinput.(vars)
    init_vars = vars[idx]

    @assert all([length(x.second) == 1 for x in init_values]) "Initialization via linear interpolation
        expected one value for the state at the end point, got $([length(x.second) for x in init_values]) values."
    init = map(init_vars) do var
        default_value = Symbolics.getdefaultval.(var)
        c = 0.0
        for _pair in init_values
            if isequal(var, _pair.first)
                c = only(_pair.second)
            end
        end
        var => vcat(default_value, c*ones(length(timepoints)-1))
    end
    return ConstantInitialization{typeof(init)}(init)
end
