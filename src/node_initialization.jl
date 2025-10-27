"""
$(TYPEDEF)

Basetype for all initialization methods for multiple shooting node variables.
"""
abstract type AbstractNodeInitialization end

"""
$(TYPEDEF)

Initializes all shooting nodes with their default value, i.e., their initial value in
the underlying problem.
```
"""
struct DefaultsInitialization <: AbstractNodeInitialization end
function (f::DefaultsInitialization)(rng::Random.AbstractRNG, layer::MultipleShootingLayer;
    params=LuxCore.setup(rng, layer), kwargs...)
    params
end

"""
$(TYPEDEF)

Initializes all shooting nodes with random values.
```
"""
struct RandomInitialization <: AbstractNodeInitialization end

"""
    (f::RandomInitialization)(rng, layer)

Initialize shooting nodes of `layer` using randomly drawn values.
"""
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

Initializes the problem using a forward solve of the problem. This results in a continuous
trajectory.
"""
struct ForwardSolveInitialization <: AbstractNodeInitialization end

"""
    (f::LinearInterpolationInitialization)(rng, layer)

Initializes shooting nodes of `layer` with values obtained via a forward integration of
the `layer` using the default values of all specified controls.
"""
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

"""
    linear_initializer(u0, u_inf, t, tspan)

Linearly interpolates u0 and u_inf for t with tspan[1] < t < tspan[2].
"""
function linear_initializer(u0, u_inf, t, tspan)
    t0, t_inf = tspan
    slope = u_inf .- u0
    val = (t - t0) ./ t_inf
    u0 .+ slope .* val
end

"""
$(TYPEDEF)

Initializes all shooting nodes with linearly-interpolated values. Linear interpolation
is calculated using the initial values of the underlying problem and the user-specified
terminal values. These are given as a Dictionary with variable indices as keys and
the corresponding terminal value.

# Fields
$(FIELDS)

# Examples
```julia-repl
julia> LinearInterpolationInitialization(Dict(1=>2.0, 2=>3.0))
LinearInterpolationInitialization{Dict{Int64, Float64}}(Dict(2 => 3.0, 1 => 2.0))
```
"""
struct LinearInterpolationInitialization{T<:AbstractDict} <: AbstractNodeInitialization
    "Terminal values for linear interpolation of initial and terminal values."
    terminal_values::T
end

"""
    (f::LinearInterpolationInitialization)(rng, layer)

Initialize shooting nodes of `layer` using linearly interpolated values between initial values
of underlying problem and terminal values given in `f.terminal_values`.
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

Initializes all shooting nodes with user-provided values. Initial values are given as
Dictionary with variable indices as keys and the corresponding vector of initial values
of adequate length.

# Fields
$(FIELDS)

# Examples
```julia-repl
julia> CustomInitialization(Dict(1=>ones(3), 2=>zeros(3)))
CustomInitialization{Dict{Int64, Vector{Float64}}}(Dict(2 => [0.0, 0.0, 0.0], 1 => [1.0, 1.0, 1.0]))
```
"""
struct CustomInitialization{I<:AbstractDict} <: AbstractNodeInitialization
    "The init values for all dependent variables"
    initial_values::I
end

"""
    (f::CustomtInitialization)(rng, layer)

Initialize shooting nodes of `layer` using custom values specified in `f.initial_values`.
"""
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

Initializes all shooting nodes using a constant value specified via the dictionary
of indices of variables and the corresponding initialization value.

# Fields
$(FIELDS)

# Examples
```julia-repl
julia> ConstantInitialization(Dict(1=>1.0, 2=>2.0))
ConstantInitialization{Dict{Int64, Float64}}(Dict(2 => 2.0, 1 => 1.0))
```
"""
struct ConstantInitialization{I<:AbstractDict} <: AbstractNodeInitialization
    "The init values for all dependent variables"
    initial_values::I
end

"""
    (f::ConstantInitialization)(rng, layer)

Initialize shooting nodes of `layer` using constant values specified in `f.initial_values`.
"""
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

"""
$(TYPEDEF)

Initializes the shooting nodes in a hybrid method.
Initialization of specific variables is done via a dictionary of variable indices of
the underlying problem and the `AbstractNodeInitialization` for their initalization.
Variables not present in the keys of `inits` are initialized using the fallback
initialization method given in `default_init`.
# Fields
$(FIELDS)

# Examples
```julia-repl
julia> HybridInitialization(Dict(1=>ConstantInitialization(Dict(1=>1.0)),
                                 2=>LinearInterpolationInitialization(Dict(2=>2.0))),
                            ForwardSolveInitialization())
HybridInitialization{Dict{Int64, Corleone.AbstractNodeInitialization}}(Dict{Int64, Corleone.AbstractNodeInitialization}(2 => LinearInterpolationInitialization{Dict{Int64, Float64}}(Dict(2 => 2.0)), 1 => ConstantInitialization{Dict{Int64, Float64}}(Dict(1 => 1.0))), ForwardSolveInitialization())
```
"""
struct HybridInitialization{P<:Dict} <: AbstractNodeInitialization
    "Dictionary of indices of variables and their corresponding initialization methods"
    inits::P
    "Fallback initialization method for variables not considered in `inits`"
    default_init::AbstractNodeInitialization
end

"""
    (f::HybridInitialization)(rng, layer)

Initialize the shooting nodes of `layer` in a hybrid method consisting of different
`AbstractNodeInitialization` methods applied to different subsets of the variables.
Variables that are not treated via the initialization methods in `f.inits` are initialized
via the fallback method `f.default_init`.
"""
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
