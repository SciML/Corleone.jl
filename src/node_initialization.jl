abstract type AbstractNodeInitialization end

"""
$(TYPEDEF)

Initializes the problem using the default values of all shooting variables.
"""
struct DefaultsInitialization <: AbstractNodeInitialization end
function (f::DefaultsInitialization)(rng::Random.AbstractRNG, layer::MultipleShootingLayer;
    params=LuxCore.setup(rng, layer), kwargs...)
    params
end
"""
$(TYPEDEF)

Initializes the problem with random values in the bounds of the variables.

# Fields
$(FIELDS)
"""
struct RandomInitialization <: AbstractNodeInitialization end

function (f::RandomInitialization)(rng::Random.AbstractRNG, layer::MultipleShootingLayer;
    params=LuxCore.setup(rng, layer),
    shooting_variables=eachindex(first(layer.layers).problem.u0))
    ps, st = params

    i = 0
    ps_rand = map(ps) do pi
        i += 1
        if i == 1
            common_variables = [i for i in shooting_variables if i in first(layer.layers).tunable_ic]
            pi.u0[common_variables] .= rand(rng, length(common_variables))
        else
            pi.u0[shooting_variables] .= rand(rng, length(shooting_variables))
        end
        pi
    end
    ps_rand, st
end

"""
$(TYPEDEF)

Initializes the problem using a single forward solve of the problem.
"""
struct ForwardSolveInitialization <: AbstractNodeInitialization end

function (f::ForwardSolveInitialization)(rng::Random.AbstractRNG, layer::MultipleShootingLayer;
    params=LuxCore.setup(rng, layer),
    shooting_variables=eachindex(first(layer.layers).problem.u0))

    ps, st = params

    u0s = [first(ps).u0]
    i = 0
    for (slayer, sps, sst) in zip(layer.layers, ps, st)
        i += 1
        common_variables = i == 1 ? [i for i in shooting_variables if i in first(layer.layers).tunable_ic] : shooting_variables
        sps.u0[common_variables] .= last(u0s)[common_variables]
        pred, _ = slayer(nothing, sps, sst)
        u0_ = pred.u[end]
        push!(u0s, u0_)
    end

    return ps, st
end


function linear_initializer(u0, u_inf, t, tspan)
    t0, t_inf = tspan
    slope = u_inf .- u0
    val = (t - t0) ./ t_inf
    u0 .+ slope .* val
end

"""
$(TYPEDEF)

Initializes the problem using a custom function which returns a vector for all variables.

# Fields
$(FIELDS)
"""
struct LinearInterpolationInitialization{T<:AbstractDict} <: AbstractNodeInitialization
    "Terminal values for linear interpolation of initial and terminal values."
    terminal_values::T
end

"""
$(FUNCTIONNAME)

Creates a (`FunctionInitialization`)[@ref] with linearly interpolates between u0 and the provided u_inf.
"""
function (f::LinearInterpolationInitialization)(rng::Random.AbstractRNG, layer::MultipleShootingLayer;
    params=LuxCore.setup(rng, layer),
    shooting_variables=eachindex(first(layer.layers).problem.u0))

    u0 = first(layer.layers).problem.u0
    @assert all([x in keys(f.terminal_values) for x in shooting_variables])
    ps, st = params
    tspan = get_tspan(layer)
    timespans = layer.shooting_intervals
    i = 0
    new_ps = map(ps) do pi
        i += 1
        if i == 1
            pi
        else
            local_tspan = timespans[i]
            interpolated_u0 = map(x -> linear_initializer(u0[x], f.terminal_values[x], first(local_tspan), tspan), shooting_variables)
            pi.u0[shooting_variables] = interpolated_u0
            pi
        end
    end
    return new_ps, st
end

"""
$(TYPEDEF)

Initializes the system with a custom vector of points provided as a Dictionary of variable => values.

# Fields
$(FIELDS)

# Note
If the variable is not present in the dictionary, we use the fallback value.
"""
struct CustomInitialization{I<:AbstractDict} <: AbstractNodeInitialization
    "The init values for all dependent variables"
    initial_values::I
end

function (f::CustomInitialization)(rng::Random.AbstractRNG, layer::MultipleShootingLayer;
    params=LuxCore.setup(rng, layer),
    shooting_variables=eachindex(first(layer.layers).problem.u0))
    ps, st = params

    i = 0
    new_ps = map(ps) do pi
        i += 1
        vari = 0
        new_u0 = map(pi.u0) do u0i
            vari += 1
            if vari ∉ shooting_variables
                u0i
            else
                f.initial_values[vari][i]
            end
        end
        pi.u0 .= new_u0
        pi
    end
    return new_ps, st
end

"""
$(TYPEDEF)

Initializes the system with a custom single value provided as a Dictionary of variable => values.

# Fields
$(FIELDS)

# Note
If the variable is not present in the dictionary, we use the fallback value.
"""
struct ConstantInitialization{I<:AbstractDict} <: AbstractNodeInitialization
    "The init values for all dependent variables"
    initial_values::I
end

function (f::ConstantInitialization)(rng::AbstractRNG, layer::MultipleShootingLayer;
    params=LuxCore.setup(rng, layer),
    shooting_variables=eachindex(first(layer.layers).problem.u0))
    ps, st = params
    new_ps = map(ps) do pi
        vari = 0
        new_u0 = map(pi.u0) do u0i
            vari += 1
            if vari ∉ shooting_variables
                u0i
            else
                f.initial_values[vari]
            end
        end
        pi.u0 .= new_u0
        pi
    end
    return new_ps, st
end

struct HybridInitialization{P<:Dict} <: AbstractNodeInitialization
    "Pair of variables and corresponding AbstractNodeInitialization methods"
    inits::P
    "Init method for remaining variables"
    default_init::AbstractNodeInitialization
end


function (f::HybridInitialization)(rng::Random.AbstractRNG, layer::MultipleShootingLayer;
    params=LuxCore.setup(rng, layer),
    shooting_variables=eachindex(first(layer.layers).problem.u0),
    kwargs...)

    ps, st = params

    defined_vars = [x.first for x in f.inits]

    forward_involved = [typeof(x.second) <: ForwardSolveInitialization for x in f.inits]
    forward_default = typeof(f.default_init) <: ForwardSolveInitialization


    any_forward = any(forward_involved) || forward_default
    forward_vars = any(forward_involved) ? reduce(vcat, defined_vars[forward_involved]) : Int64[]

    defined_vars = reduce(vcat, defined_vars)
    remaining_vars = [i for i in shooting_variables if i ∉ defined_vars]

    forward_vars = forward_default ? vcat(forward_vars, remaining_vars) : forward_vars

    init_copy = copy(f.inits)
    init_copy = any_forward ? delete!(init_copy, ForwardSolveInitialization()) : init_copy

    init_copy = begin
        if forward_default
            init_copy
        else
            merge(init_copy, Dict(remaining_vars => f.default_init))
        end
    end

    for p in init_copy
        ps, st = p.second(rng, layer; shooting_variables=p.first, params=(ps, st))
    end
    ps, st = begin
        if any_forward
            ForwardSolveInitialization()(rng, layer; shooting_variables=forward_vars, params=(ps, st))
        else
            ps, st
        end
    end

    return ps, st
end
