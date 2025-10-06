struct ControlParameter{T, C, B}
    name::Symbol
    "The timepoints"
    t::T
    "The initial control parameters. Either a vector or a function (rng,t,bounds) -> u "
    controls::C
    "The bounds as a tuple"
    bounds::B
end

default_u(rng, t, bounds) = zeros(eltype(t), size(t))
default_bounds(t::AbstractVector{T}) where T<:Real = (fill(typemin(T), size(t)), fill(typemax(T), size(t)))

function ControlParameter(t::AbstractVector; name::Symbol=gensym(:w), controls=default_u, bounds=default_bounds)
    ControlParameter{typeof(t),typeof(controls),typeof(bounds)}(name, t, controls, bounds)
end

function restrict_controls(c::Tuple, lo, hi)
    map(c) do ci
        restrict_controls(ci, lo, hi)
    end
end

function restrict_controls(c::ControlParameter, lo, hi)
    idx = findall(lo .<= c.t .< hi)
    controls = c.controls == default_u ? c.controls : c.controls[idx]
    bounds = c.bounds == default_bounds ? c.bounds : (length(c.bounds[1]) == 1 ? c.bounds : (c.bounds[1][idx], c.bounds[2][idx]))
    return ControlParameter(copy(c.t[idx]), name = c.name, controls = controls, bounds = bounds)
end

function unrestrict_controls(c::Tuple, tspan)
    map(c) do ci
        unrestrict_controls(ci, tspan)
    end
end

function unsrestrict_controls(c::ControlParameter, tspan)
    dt = first(diff(c.t))
    timegrid = first(tspan):dt:last(tspan)
    n_repeat = Int(length(timegrid)/length(c.t))
    new_bounds = begin
        if length(c.bounds[1]==1)
            c.bounds
        else
            (c.bounds[1][1], c.bounds[2][1])
        end
    end
    return ControlParameter(timegrid, name=c.name, controls=repeat(c.controls, n_repeat), bounds=new_bounds)
end

get_timegrid(parameters::ControlParameter) = collect(parameters.t)
get_controls(::Random.AbstractRNG, parameters::ControlParameter{<:Any, <:AbstractArray}) = deepcopy(parameters.controls)
get_controls(rng::Random.AbstractRNG, parameters::ControlParameter{<:Any, <:Function}) = parameters.controls(rng, parameters.t, parameters.bounds)
get_bounds(parameters::ControlParameter{<:Any, <:Any, <:Tuple}) = begin
    _bounds = getfield(parameters, :bounds)
    nc = length(parameters.t)
    if length(_bounds[1]) == length(_bounds[2]) == 1
        return (repeat([_bounds[1]], nc), repeat([_bounds[2]], nc))
    elseif length(_bounds[1]) == length(_bounds[2]) == nc
        return _bounds
    else
        throw("Incompatible control bound definition. Got $(length(_bounds[1])) elements, expected $nc.")
    end
end
get_bounds(parameters::ControlParameter{<:Any, <:Any, <:Function}) = parameters.bounds(parameters.t)

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

function build_index_grid(controls::ControlParameter...; offset::Bool=true, tspan::Tuple=(-Inf, Inf), subdivide::Int64 = typemax(Int64))
    ts = map(controls) do ci
        clamp.(get_timegrid(ci), tspan...)
    end
    time_grid = vcat(reduce(vcat, ts), collect(tspan)) |> sort! |> unique! |> Base.Fix1(filter!, isfinite)
    indices = zeros(Int64, length(ts), size(time_grid, 1)-1)
    for i in axes(indices, 1), j in axes(indices, 2)
        indices[i,j] = clamp(
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
    N = size(indices,2)
    if N > subdivide
        ranges = get_subvector_indices(N, subdivide)
        return Tuple(indices[:, i] for i in ranges)
    end
    return indices
end

function collect_tspans(controls::ControlParameter...; tspan=(-Inf, Inf), subdivide::Int64 = typemax(Int64))
    ts = map(controls) do ci
        clamp.(get_timegrid(ci), tspan...)
    end
    time_grid = vcat(reduce(vcat, ts), collect(tspan)) |> sort! |> unique! |> Base.Fix1(filter!, isfinite)
    fullgrid = collect(ti for ti in zip(time_grid[1:end-1], time_grid[2:end]))
    N = size(fullgrid,1)
    if N > subdivide
        ranges = get_subvector_indices(N, subdivide)
        return Tuple(tuple(fullgrid[i]...) for i in ranges)
    end
    tuple(fullgrid...)
end

function collect_local_controls(rng, controls::ControlParameter...)
    reduce(vcat, map(vec ∘ Base.Fix1(get_controls, rng), controls))
end

function collect_local_control_bounds(lower::Bool, controls::ControlParameter...)
    reduce(vcat, map(vec ∘ (lower ? first : last) ∘ get_bounds, controls))
end
