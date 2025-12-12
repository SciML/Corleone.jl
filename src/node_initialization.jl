"""
$(SIGNATURES)

Initializes all shooting nodes with random values.

# Arguments
- `rng::Random.AbstractRNG` a random number generator 
- `shooting::MultipleShootingLayer` a shooting layer

# Keyworded Arguments 
- `ps` the default parameters of the `shooting` layer. 
"""
function random_initialization(
    rng::Random.AbstractRNG,
    shooting::MultipleShootingLayer;
    ps=default_initialization(rng, shooting),
    kwargs...,
)
    (; layer) = shooting
    (; tunable_ic) = layer
    u0 = last(ps).u0
    isempty(u0) && return ps
    lb, ub = get_bounds(shooting)
    vals = map(enumerate(zip(ps, lb, ub))) do (i, (plocal, lbi, ubi))
        unew = if i == 1
            _random_value(rng, lbi.u0[tunable_ic], ubi.u0[tunable_ic])
        else
            _random_value(rng, lbi.u0, ubi.u0)
        end
        merge(plocal, (; u0=unew))
    end
    return NamedTuple{keys(ps)}(vals)
end

"""
$(SIGNATURES)

Linearly interpolates u0 and u_inf for t with tspan[1] < t < tspan[2].

# Arguments
- `rng::Random.AbstractRNG` a random number generator 
- `shooting::MultipleShootingLayer` a shooting layer

# Keyworded Arguments 
- `ps` the default parameters of the `shooting` layer. 
- `u_infinity::AbstractArray` the value at the last timepoint for all states.
"""
function linear_initialization(
    rng::Random.AbstractRNG,
    shooting::MultipleShootingLayer;
    ps=default_initialization(rng, shooting),
    u_infinity=last(ps).u0,
    kwargs...,
)
    (; shooting_intervals, layer) = shooting
    (; problem, tunable_ic) = layer
    isempty(u_infinity) && return ps
    u0 = first(ps).u0
    if isempty(u0)
        u0 = problem.u0
    end
    t0, tinf = problem.tspan
    foreach(enumerate(ps)) do (i, sps)
        t = shooting_intervals[i][1]
        u0new = (u_infinity .- u0) .* (t - t0) ./ tinf .+ u0
        sps.u0 .= i > 1 ? u0new : u0new[tunable_ic]
    end
    return ps
end

__setu0!(u0::AbstractArray, ::Any...) = u0

function __setu0!(u0::AbstractArray, unew::AbstractArray, tunable_ics)
    return unew[tunable_ics]
end

function __setu0!(u0::AbstractArray, unew::Dict, tunable_ics)
    map(tunable_ics) do i
        get(unew, i, u0[i])
    end
end

function __setu0!(u0::AbstractArray, unew::Base.AbstractVecOrTuple{<:Pair}, tunable_ics)
    map(tunable_ics) do i
        id = findfirst(==(i), first.(unew))
        isnothing(id) && u0[i]
        last.(unew)[id]
    end
end

"""
$(SIGNATURES)


Initializes all shooting nodes with user-provided values. Initial values are given as an iterable collection of 
`AbstractArray`s, `Dict`s, or a vector of `Pair`s.  The variable indices are interpretated depending on the passed value

- `AbstractArray` just sets the initial condition with the provided value. This assumes equal dimensionality.
- `Dict` assumes the keys represent indices of the initial condition. 
- A `Vector` or `Tuple` of `Pair`s assumes the first value represents an index of the initial condition.

Other options simply skip the corresponding interval.

# Arguments
- `rng::Random.AbstractRNG` a random number generator 
- `shooting::MultipleShootingLayer` a shooting layer

# Keyworded Arguments 
- `ps` the default parameters of the `shooting` layer. 
- `u0s` the collection of initial condtions.  

# Example 
```julia 
# Assumes a shooting layer with 3 intervals 
# Skips the first interval 
# Sets the first index of the second interval to 3.0 
# Sets the second index of the third interval to 5.0 and the 5th index to -1.0
custom_initialization(rng, shooting, u0s = [nothing, Dict(1 => 3.0), (2 => 5.0, 5 => -1.0)])
``` 
"""
function custom_initialization(
    rng::Random.AbstractRNG,
    shooting::MultipleShootingLayer;
    ps=default_initialization(rng, shooting),
    u0s=[],
    kwargs...,
)
    (; layer) = shooting
    (; tunable_ic) = layer
    isempty(u0s) && return ps
    foreach(enumerate(ps)) do (i, sps)
        if lastindex(u0s) >= i
            sps.u0 .= __setu0!(sps.u0, u0s[i], i > 1 ? eachindex(sps.u0) : tunable_ic)
        end
    end
    return ps
end

"""
$(SIGNATURES)

Initializes the problem using a forward solve of the problem. This results in a continuous
trajectory.

# Arguments
- `rng::Random.AbstractRNG` a random number generator 
- `shooting::MultipleShootingLayer` a shooting layer

# Keyworded Arguments 
- `ps` the default parameters of the `shooting` layer. 
"""
function forward_initialization(
    rng::Random.AbstractRNG,
    shooting::MultipleShootingLayer;
    ps=default_initialization(rng, shooting),
	fixed_indices = Int[],
    kwargs...,
)
    (; layer) = shooting
    st = LuxCore.initialstates(rng, shooting)
    u0s = [first(ps).u0]
	for (i,(sps, sst)) in enumerate(zip(ps, st))
        u0 = i ∈ fixed_indices ? sps.u0 : last(u0s)
        sps.u0 .= u0[eachindex(sps.u0)]
        pred, _ = layer(nothing, sps, sst)
        u0_ = pred.u[end]
        push!(u0s, u0_)
    end
    return ps
end

"""
$(SIGNATURES)


Initializes all shooting nodes with user-provided value. The variable indices are interpretated depending on the passed value

- `AbstractArray` just sets the initial condition with the provided value. This assumes equal dimensionality.
- `Dict` assumes the keys represent indices of the initial condition. 
- A `Vector` or `Tuple` of `Pair`s assumes the first value represents an index of the initial condition.

Other options simply skip the corresponding interval.

# Arguments
- `rng::Random.AbstractRNG` a random number generator 
- `shooting::MultipleShootingLayer` a shooting layer

# Keyworded Arguments 
- `ps` the default parameters of the `shooting` layer. 
- `u0` the initial condtion.  

# Example 
```julia 
# Assumes a shooting layer with 3 intervals 
# Sets the first index of all intervals to 3.0 
constant_initialization(rng, shooting, u0 = Dict(1 => 3.0))
constant_initialization(rng, shooting, u0 = (1 => 3.0,))
# Set the initial condition to the given vector
constant_initialization(rng, shooting, u0 = [1., 2., 4.])
``` 
"""
function constant_initialization(
    rng::Random.AbstractRNG,
    shooting::MultipleShootingLayer;
    ps=default_initialization(rng, shooting),
    u0=nothing,
    kwargs...,
)
    (; layer) = shooting
    (; tunable_ic) = layer
    isempty(u0) && return ps
    foreach(enumerate(ps)) do (i, sps)
        sps.u0 .= __setu0!(sps.u0, u0, i > 1 ? eachindex(sps.u0) : tunable_ic)
    end
    return ps
end

"""
$(SIGNATURES)

Initializes the shooting nodes in a hybrid method by applying the provided methods and indices `f` 
sequentially. Here we assume the structure `interval => method` for the initialization.

# Arguments
- `rng::Random.AbstractRNG` a random number generator 
- `shooting::MultipleShootingLayer` a shooting layer
- `f::Pair`s of the interval index and applied method

# Keyworded Arguments 
- `ps` the default parameters of the `shooting` layer. 

All other keyworded arguments are passed on to the functions below. 
"""
function hybrid_initialization(
    rng::Random.AbstractRNG,
    shooting::MultipleShootingLayer,
    f::Pair...;
    ps=default_initialization(rng, shooting),
    kwargs...,
)
	(intervals, method), rest = Base.first(f), Base.tail(f)
	pnew = method(rng, shooting; ps = deepcopy(ps), kwargs...)
	names = ntuple(i->keys(ps)[intervals[i]],size(intervals,1))
	@info names
	pnew = merge(ps, NamedTuple{names}(pnew))
    return hybrid_initialization(rng, shooting, rest...; ps=pnew, kwargs...)
end

function hybrid_initialization(
    rng::Random.AbstractRNG,
    shooting::MultipleShootingLayer;
    ps=default_initialization(rng, shooting),
    kwargs...,
)
    return ps
end

#=
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

=#
