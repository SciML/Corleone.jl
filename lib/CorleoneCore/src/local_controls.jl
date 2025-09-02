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
default_bounds(t::AbstractVector{T}) where T <: Real = (fill(typemin(T), size(t)), fill(typemax(T), size(t)))

function ControlParameter(t::AbstractVector; name::Symbol = gensym(:w), controls = default_u, bounds = default_bounds)
    ControlParameter{typeof(t), typeof(controls), typeof(bounds)}(name, t, controls, bounds)
end

get_timegrid(parameters::ControlParameter) = collect(parameters.t)
get_controls(::Random.AbstractRNG, parameters::ControlParameter{<:Any, <:AbstractArray}) = deepcopy(parameters.controls)
get_controls(rng::Random.AbstractRNG, parameters::ControlParameter{<:Any, <:Function}) = parameters.controls(rng, parameters.t, parameters.bounds)
get_bounds(parameters::ControlParameter{<:Any, <:Any, <:Tuple}) = getfield(parameters, :bounds)
get_bounds(parameters::ControlParameter{<:Any, <:Any, <:Function}) = parameters.bounds(parameters.t) 

function check_consistency(rng::Random.AbstractRNG, parameters::ControlParameter)
    grid = get_timegrid(parameters)
    u = get_controls(rng, parameters)
    lb, ub = get_bounds(parameters)
    @assert issorted(grid) "Time grid is not sorted."
    @assert get_timegrid(parameters) == unique(grid) "Time grid is not unique."
    @assert all(lb .<= ub) "Bounds are inconsistent"
    @assert size(lb) == size(ub) == size(u) == size(grid) "Sizes are inconsistent"
    @assert all(lb .<= u .<= ub)  "Initial values are inconsistent"
end

function build_index_grid(controls::ControlParameter...; offset::Bool = true)
    ts = map(get_timegrid, controls)
    time_grid = reduce(vcat, ts) |> sort! |> unique!
    indices = zeros(Int64, length(ts), size(time_grid, 1))
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
    return indices
end

function collect_tspans(controls::ControlParameter...)
    ts = map(get_timegrid, controls)
    time_grid = reduce(vcat, ts) |> sort! |> unique!
    tuple(collect(ti for ti in zip(time_grid[1:end-1], time_grid[2:end]))...)
end

function collect_local_controls(rng, controls::ControlParameter...)
    reduce(vcat, map(vec ∘ Base.Fix1(get_controls, rng), controls))
end