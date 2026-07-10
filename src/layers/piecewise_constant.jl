"""
$(TYPEDEF)

A single piecewise-constant control input described by a sorted breakpoint grid
`tpoints` and an initial value vector `values`.

`tpoints` is baked in at construction (static structure); `values` seeds
`LuxCore.initialparameters` to produce the optimizer decision vector
`ps.controls[i]` for this control input.

# Fields
$(FIELDS)

"""

@concrete struct PiecewiseParameter <: LuxCore.AbstractLuxLayer
    "Parameter identifier"
    parameter_id
    "Initializer for the control at the breakpoint grid"
    init
    "Sorted breakpoint grid"
    tpoints
    "Bounds"
    bounds
    "Injected points for shooting"
    injected::Vector{Int64}
end

function PiecewiseParameter(parameter_id, tpoints::AbstractVector{<:Number}, init=nothing, bounds=nothing) where {P,V,T}
    @assert issorted(tpoints) "tpoints must be sorted"
    @assert allunique(tpoints) "tpoints must be unique"
    return PiecewiseParameter(parameter_id, init, tpoints, bounds, Int64[])
end

function reset!(pc::PiecewiseParameter)
    deleteat!(pc.tpoints, pc.injected)
    empty!(pc.injected)
    return pc
end

function inject!(pc::PiecewiseParameter, t::T) where T<:Number
    idx = searchsortedlast(pc.tpoints, t)
    if pc.tpoints[idx] != t
        insert!(pc.tpoints, idx+1, t)
        push!(pc.injected, idx+1)
    end
    return pc
end

LuxCore.display_name(pc::PiecewiseParameter) = begin
    x = pc.parameter_id
    SymbolicIndexingInterface.hasname(x) && return SymbolicIndexingInterface.getname(x)
    Symbol(x)
end

get_ps_index(sys, x) =
    if isa(SymbolicIndexingInterface.symbolic_type(x), SymbolicIndexingInterface.NotSymbolic)
        return x
    else
        return SymbolicIndexingInterface.parameter_index(sys, x)
    end

get_ps_index(sys, x::Base.AbstractVecOrTuple) = map(x) do xi
    get_ps_index(sys, xi)
end

function get_parameter_index(container::C, pc::PiecewiseParameter) where C
    (; parameter_id) = pc
    get_ps_index(container, parameter_id)
end

get_parameter_index(::Nothing, pc::PiecewiseParameter) = begin
    (; parameter_id) = pc
    @assert isa(SymbolicIndexingInterface.symbolic_type(parameter_id), SymbolicIndexingInterface.NotSymbolic) "Symbolic indices are only valid when providing a symbolic container!"
    parameter_id
end

get_parameter_shape(pc::PiecewiseParameter) = begin
    (; parameter_id) = pc
    isa(parameter_id, Symbol) ? (1,) : size(collect(parameter_id))
end

function get_lower_bound(pc::PiecewiseParameter, ps, st)
    (; bounds) = pc
    isnothing(bounds) && return get_lower_bound(ps)
    first_or_first(bounds, ps, st)
end

function get_upper_bound(pc::PiecewiseParameter, ps, st)
    (; bounds) = pc
    isnothing(bounds) && return get_lower_bound(ps)
    last_or_last(bounds, ps, st)
end

LuxCore.initialparameters(rng::Random.AbstractRNG, pc::PiecewiseParameter{<:Any,<:Base.Callable}) = pc.init(rng, eltype(pc.tpoints), length(pc.tpoints)+1)
LuxCore.initialparameters(::Random.AbstractRNG, pc::PiecewiseParameter{<:Any,Nothing}) = begin
    (; tpoints, parameter_id) = pc
    shape = get_parameter_shape(pc)
    fill(zeros(eltype(tpoints), shape...), length(tpoints)+1)
end

LuxCore.parameterlength(pc::PiecewiseParameter) = (length(pc.tpoints)+1) * max(1, prod(get_parameter_shape(pc)))

LuxCore.initialstates(::Random.AbstractRNG, pc::PiecewiseParameter) = (;
    tpoints=pc.tpoints,
    current_index=firstindex(pc.tpoints),
    first_index=firstindex(pc.tpoints),
    last_index=lastindex(pc.tpoints),
    cache=nothing
)

LuxCore.statelength(::PiecewiseParameter) = length(pc.tpoints) + 4

using SparseArrays

function collect_activity_pattern(timepoints::AbstractVector, pc::PiecewiseParameter, ps, st)
    (; tpoints) = pc
    N = prod(get_parameter_shape(pc))
    idx = searchsortedlast.(Ref(tpoints), timepoints) .+ 1
    M = idx .== transpose(1:(length(tpoints)+1))
    M |> sparse
end

function collect_active_parameters(timepoints::AbstractVector, pc::PiecewiseParameter, ps, st)
    M = collect_activity_pattern(timepoints, pc, ps, st)
    kron(M, ones(Bool, 1, N))  
end

function active_parameters(a, b)
    m, n = length(a), length(b)
    I, J = Int[], Int[]
    for i in 1:m
        hi = i < m ? a[i+1] : Inf 
        jlo = searchsortedlast(b, a[i]) + 1 
        jhi = searchsortedfirst(b, hi)
        for j in jlo:jhi 
            push!(I, i)
            push!(J, j)
        end
    end
    sparse(I, J, true, m, n + 1)
end

function find_index!(::Nothing, t, st)
    t_active = only(t)
    tpoints = st.tpoints
    idx = searchsortedlast(tpoints, t_active)
    return idx + 1
end

find_index!(d::AbstractDict, t, st) = get!(d, t) do t
    find_index!(nothing, t, st)
end

@non_differentiable find_index!(cache, t, st)

function (pc::PiecewiseParameter)(t::T, ps, st) where T<:Number
    idx = find_index!(st.cache, t, st)
    return ps[idx], merge(st, (; current_index=idx))
end

number_of_shooting_constraints(pc::PiecewiseParameter) = size(pc.injected, 1)

function shooting_constraints(pc::PiecewiseParameter, ps, st)
    (; injected) = pc
    return ps[injected] .- ps[injected .- 1]
end

function shooting_constraints!(res::AbstractArray, pc::PiecewiseParameter, ps, st)
    (; injected) = pc
    res .= ps[injected] .- ps[injected .- 1]
end



#=
"""
$(SIGNATURES)

Constructs a `ControlParameter` with piecewise constant discretizations introduced at
timepoints `t`. Optionally

```julia-repl
julia> ControlParameter(0:1.0:4.0, name=:c)
ControlParameter{StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}, typeof(Corleone.default_u), typeof(Corleone.default_bounds)}(:c, 0.0:1.0:10.0, Corleone.default_u, Corleone.default_bounds)
```

```julia-repl
julia> ControlParameter(0.0:1.0:4.0, name=:c, controls = zeros(5))
ControlParameter{StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}, Vector{Float64}, typeof(Corleone.default_bounds)}(:c, 0.0:1.0:4.0, [0.0, 0.0, 0.0, 0.0, 0.0], Corleone.default_bounds)
```


```julia-repl
julia> ControlParameter(0:1.0:9.0, name=:c1, controls=zeros(5), bounds=(0.0,1.0))
ControlParameter{StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}, Vector{Float64}, Tuple{Float64, Float64}}(:c1, 0.0:1.0:9.0, [0.0, 0.0, 0.0, 0.0, 0.0], (0.0, 1.0))
```
The latter is functionally equivalent to the following example, specifying all bounds individually:
```julia-repl
julia> ControlParameter(0:1.0:9.0, name=:c1, controls=zeros(5), bounds=(zeros(5),ones(5)))
ControlParameter{StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}, Vector{Float64}, Tuple{Vector{Float64}, Vector{Float64}}}(:c1, 0.0:1.0:9.0, [0.0, 0.0, 0.0, 0.0, 0.0], ([0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0]))
```
"""
function ControlParameter(t::AbstractVector; name::Symbol=gensym(:w), controls=default_u, bounds=default_bounds)
    return ControlParameter{typeof(t),typeof(controls),typeof(bounds)}(name, t, controls, bounds)
end

get_timegrid(parameters::ControlParameter, tspan=(-Inf, Inf)) = begin
    (; t) = parameters
    idx = isnothing(tspan) ? eachindex(t) : findall(tspan[1] .<= t .< tspan[2])
    t[idx]
end

```
$(SIGNATURES)

Computes the number of discretized controls restricted to given `tspan`.
```
function control_length(parameters::ControlParameter; tspan=nothing, kwargs...)
    (; t) = parameters
    idx = isnothing(tspan) ? eachindex(t) : findall(tspan[1] .<= t .< tspan[2])
    return size(idx, 1)
end

```
$(SIGNATURES)

Returns discretized controls of ControlParameter `params` restricted to given `tspan`.
```
function get_controls(::Random.AbstractRNG, parameters::ControlParameter{<:Any,<:AbstractArray}; raw=false, tspan=nothing, kwargs...)
    (; t, controls) = parameters
    raw && return controls
    idx = isnothing(tspan) ? eachindex(t) : findall(tspan[1] .<= t .< tspan[2])
    return controls[idx]
end

function get_controls(rng::Random.AbstractRNG, parameters::ControlParameter{<:Any,<:Function}; raw=false, tspan=nothing, kwargs...)
    (; t) = parameters
    bounds = get_bounds(parameters; tspan, kwargs...)
    idx = isnothing(tspan) ? eachindex(t) : findall(tspan[1] .<= t .< tspan[2])
    return parameters.controls(rng, t[idx], bounds)
end

```
$(SIGNATURES)

Returns bounds of discretized controls restricted to given `tspan`.
```
function get_bounds(parameters::ControlParameter{<:Any,<:Any,<:Tuple}; tspan=nothing, kwargs...)
    (; t) = parameters
    idx = isnothing(tspan) ? eachindex(t) : findall(tspan[1] .<= t .< tspan[2])
    nc = size(idx, 1)
    _bounds = parameters.bounds
    if length(_bounds[1]) == length(_bounds[2]) == 1
        return (repeat([_bounds[1]], nc), repeat([_bounds[2]], nc))
    elseif length(_bounds[1]) == length(_bounds[2]) == length(t)
        return (_bounds[1][idx], _bounds[2][idx])
    end
    throw("Incompatible control bound definition. Got $(length(_bounds[1])) elements, expected $(length(t)).")
end

get_bounds(parameters::ControlParameter{<:Any,<:Any,<:Function}; tspan=(-Inf, Inf), kwargs...) = parameters.bounds(get_timegrid(parameters, tspan; kwargs...))

function check_consistency(rng::Random.AbstractRNG, parameters::ControlParameter)
    grid = get_timegrid(parameters)
    u = get_controls(rng, parameters; raw=true)
    lb, ub = get_bounds(parameters)
    @assert issorted(grid) "Time grid is not sorted."
    @assert get_timegrid(parameters) == unique(grid) "Time grid is not unique."
    @assert all(lb .<= ub) "Bounds are inconsistent"
    @assert size(lb) == size(ub) == size(u) == size(grid) "Sizes are inconsistent"
    return @assert all(lb .<= u .<= ub) "Initial values are inconsistent"
end

function get_subvector_indices(M::Int, L::Int)
    # Handle invalid inputs
    if M < 0 || L <= 0
        error("M must be a non-negative integer and L must be a positive integer.")
    end

    # Calculate the number of full-length vectors (N)
    N = floor(Int, M / L)

    indices = Vector{UnitRange{Int64}}()

    # Create the N full-length vectors
    for i in 1:N
        start_idx = (i - 1) * L + 1
        end_idx = i * L
        push!(indices, start_idx:end_idx)
    end

    # Create the last vector with remaining elements
    remaining_length = M - N * L
    if remaining_length > 0
        last_start_idx = N * L + 1
        last_end_idx = M
        push!(indices, last_start_idx:last_end_idx)
    end

    return indices
end

function build_index_grid(controls::ControlParameter...; offset::Bool=true, tspan::Tuple=(-Inf, Inf), subdivide::Int64=typemax(Int64))
    ts = map(controls) do ci
        get_timegrid(ci, tspan)
    end
    time_grid = vcat(reduce(vcat, ts), collect(tspan)) |> sort! |> unique! |> Base.Fix1(filter!, isfinite)
    indices = zeros(Int64, length(ts), size(time_grid, 1) - 1)
    for i in axes(indices, 1), j in axes(indices, 2)
        indices[i, j] = clamp(
            searchsortedlast(ts[i], time_grid[j]),
            firstindex(ts[i]), lastindex(ts[i])
        )
    end
    # Offset
    if offset
        for i in axes(indices, 1)
            if i > 1
                indices[i, :] .+= maximum(indices[i-1, :])
            end
        end
    end
    # Check the gridsize
    N = size(indices, 2)
    # Normalize for the first index here
    indices .-= minimum(indices) - 1
    if N > subdivide
        ranges = get_subvector_indices(N, subdivide)
        return Tuple(indices[:, i] for i in ranges)
    end
    return indices
end

find_shooting_indices(tspan, control::ControlParameter) = any(first(tspan) .== control.t)


function collect_tspans(controls::ControlParameter...; tspan=(-Inf, Inf), subdivide::Int64=typemax(Int64))
    ts = map(controls) do ci
        get_timegrid(ci, tspan)
    end
    time_grid = vcat(reduce(vcat, ts), collect(tspan)) |> sort! |> unique! |> Base.Fix1(filter!, isfinite)
    fullgrid = collect(ti for ti in zip(time_grid[1:(end-1)], time_grid[2:end]))
    N = size(fullgrid, 1)
    if N > subdivide
        ranges = get_subvector_indices(N, subdivide)
        return Tuple(tuple(fullgrid[i]...) for i in ranges)
    end
    return tuple(fullgrid...)
end

```
$(SIGNATURES)
Collect all discretized `controls` in a flat vector.
```
function collect_local_controls(rng, controls::ControlParameter...; kwargs...)
    return reduce(
        vcat, map(controls) do control
            get_controls(rng, control; kwargs...)
        end
    )
end

```
$(SIGNATURES)
Collect all lower and upper bounds of discretized `controls` in flat vectors.
```
function collect_local_control_bounds(controls::ControlParameter...; kwargs...)
    bounds = map(controls) do control
        get_bounds(control; kwargs...)
    end
    lb = reduce(vcat, first.(bounds))
    ub = reduce(vcat, last.(bounds))
    return lb, ub
end
=#