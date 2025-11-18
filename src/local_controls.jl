"""
$(TYPEDEF)
Implements a piecewise constant control discretization.

# Fields
$(FIELDS)
"""
struct ControlParameter{T,C,B}
  "The name of the control"
  name::Symbol
  "The timepoints at which discretized variables are introduced"
  t::T
  "The initial values for the controls. Either a vector or a function (rng,t,bounds) -> u"
  controls::C
  "The bounds as a tuple"
  bounds::B
end

default_u(rng, t, bounds) = zeros(eltype(t), size(t))
default_bounds(t::AbstractVector{T}) where {T<:Real} = (fill(typemin(T), size(t)), fill(typemax(T), size(t)))

"""
$(METHODLIST)

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
  ControlParameter{typeof(t),typeof(controls),typeof(bounds)}(name, t, controls, bounds)
end

"""
Computes and returns a new tuple of `ControlParameter` with their timespans reduced to `(lo,hi)`.
"""
function restrict_controls(c::Tuple, lo, hi)
  map(c) do ci
    restrict_controls(ci, lo, hi)
  end
end

"""
Computes and returns a new `ControlParameter` from `c` with timepoints restricted to
`lo` .< `c.t` .< `hi`.
"""
function restrict_controls(c::ControlParameter, lo, hi)
  idx = findall(lo .<= c.t .< hi)
  controls = c.controls == default_u ? c.controls : c.controls[idx]
  bounds = c.bounds == default_bounds ? c.bounds : (length(c.bounds[1]) == 1 ? c.bounds : (c.bounds[1][idx], c.bounds[2][idx]))
  return ControlParameter(copy(c.t[idx]), name=c.name, controls=controls, bounds=bounds)
end

get_timegrid(parameters::ControlParameter, tspan=(-Inf, Inf)) = begin
  (; t) = parameters
  idx = isnothing(tspan) ? eachindex(t) : findall(tspan[1] .<= t .< tspan[2])
  t[idx]
end

function control_length(parameters::ControlParameter, tspan=nothing; kwargs...)
  (; t) = parameters
  idx = isnothing(tspan) ? eachindex(t) : findall(tspan[1] .<= t .< tspan[2])
  size(idx, 1)
end

function get_controls(::Random.AbstractRNG, parameters::ControlParameter{<:Any,<:AbstractArray}; tspan=nothing, kwargs...)
  (; t, controls) = parameters
  idx = isnothing(tspan) ? eachindex(t) : findall(tspan[1] .<= t .< tspan[2])
  controls[idx]
end

function get_controls(rng::Random.AbstractRNG, parameters::ControlParameter{<:Any,<:Function}; tspan=nothing, kwargs...)
  (; t, bounds) = parameters
  idx = isnothing(tspan) ? eachindex(t) : findall(tspan[1] .<= t .< tspan[2])
  bounds = length(bounds[1]) == length(t) ? (bounds).([idx]) : bounds
  parameters.controls(rng, t[idx], bounds)
end

function get_bounds(parameters::ControlParameter{<:Any,<:Any,<:Tuple}; kwargs...) 
  nc = control_length(parameters; kwargs...) 
  _bounds = parameters.bounds
  if length(_bounds[1]) == length(_bounds[2]) == 1
    return (repeat([_bounds[1]], nc), repeat([_bounds[2]], nc))
  elseif length(_bounds[1]) == length(_bounds[2]) == nc
    return _bounds
  else
    throw("Incompatible control bound definition. Got $(length(_bounds[1])) elements, expected $nc.")
  end
end

get_bounds(parameters::ControlParameter{<:Any,<:Any,<:Function}; kwargs...) = parameters.bounds(get_timegrid(parameters; kwargs...))

function check_consistency(rng::Random.AbstractRNG, parameters::ControlParameter)
  grid = get_timegrid(parameters)
  u = get_controls(rng, parameters)
  lb, ub = get_bounds(parameters)
  @assert issorted(grid) "Time grid is not sorted."
  @assert get_timegrid(parameters) == unique(grid) "Time grid is not unique."
  @assert all(lb .<= ub) "Bounds are inconsistent"
  @assert size(lb) == size(ub) == size(u) == size(grid) "Sizes are inconsistent"
  @assert all(lb .<= u .<= ub) "Initial values are inconsistent"
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
  fullgrid = collect(ti for ti in zip(time_grid[1:end-1], time_grid[2:end]))
  N = size(fullgrid, 1)
  if N > subdivide
    ranges = get_subvector_indices(N, subdivide)
    return Tuple(tuple(fullgrid[i]...) for i in ranges)
  end
  tuple(fullgrid...)
end

function collect_local_controls(rng, controls::ControlParameter...; kwargs...)
  reduce(vcat, map(controls) do control
    get_controls(rng, control; kwargs...)
  end)
end

function collect_local_control_bounds(controls::ControlParameter...; kwargs...)
   bounds = map(controls) do control 
		get_bounds(control; kwargs...)
	end 
	lb = reduce(vcat, first.(bounds)) 
	ub = reduce(vcat, last.(bounds)) 
	lb, ub
end
