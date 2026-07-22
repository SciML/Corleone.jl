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

function PiecewiseParameter(parameter_id, tpoints::AbstractVector{<:Number}, init = nothing, bounds = nothing)
    @assert issorted(tpoints) "tpoints must be sorted"
    @assert allunique(tpoints) "tpoints must be unique"
    return PiecewiseParameter(parameter_id, init, tpoints, bounds, Int64[])
end

function reset!(pc::PiecewiseParameter)
    deleteat!(pc.tpoints, sort!(unique!(pc.injected)))
    empty!(pc.injected)
    return pc
end

function inject!(pc::PiecewiseParameter, t::T) where {T <: Number}
    idx = searchsortedlast(pc.tpoints, t)
    if pc.tpoints[idx] != t
        insert!(pc.tpoints, idx + 1, t)
        # Update all
        idxs = findall(pc.injected .> idx + 1)
        pc.injected[idxs] .+= 1
        push!(pc.injected, idx + 1)
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

function get_parameter_index(container::C, pc::PiecewiseParameter) where {C}
    (; parameter_id) = pc
    return get_ps_index(container, parameter_id)
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
    return first_or_first(bounds, ps, st)
end

function get_upper_bound(pc::PiecewiseParameter, ps, st)
    (; bounds) = pc
    isnothing(bounds) && return get_upper_bound(ps)
    return last_or_last(bounds, ps, st)
end

LuxCore.initialparameters(rng::Random.AbstractRNG, pc::PiecewiseParameter{<:Any, <:Base.Callable}) = pc.init(rng, eltype(pc.tpoints), length(pc.tpoints) + 1)
LuxCore.initialparameters(::Random.AbstractRNG, pc::PiecewiseParameter{<:Any, Nothing}) = begin
    (; tpoints, parameter_id) = pc
    shape = get_parameter_shape(pc)
    fill(zeros(eltype(tpoints), shape...), length(tpoints) + 1)
end

LuxCore.parameterlength(pc::PiecewiseParameter) = (length(pc.tpoints) + 1) * max(1, prod(get_parameter_shape(pc)))

LuxCore.initialstates(::Random.AbstractRNG, pc::PiecewiseParameter) = (;
    tpoints = pc.tpoints,
    current_index = firstindex(pc.tpoints),
    first_index = firstindex(pc.tpoints),
    last_index = lastindex(pc.tpoints),
    cache = nothing,
)

LuxCore.statelength(::PiecewiseParameter) = length(pc.tpoints) + 4

function collect_activity_pattern(timepoints::AbstractVector, pc::PiecewiseParameter, ps, st)
    (; tpoints) = pc
    N = prod(get_parameter_shape(pc))
    idx = searchsortedlast.(Ref(tpoints), timepoints) .+ 1
    M = idx .== transpose(1:(length(tpoints) + 1))
    return M |> sparse
end

function find_index!(::Nothing, t, st)
    t_active = only(t)
    tpoints = st.tpoints
    idx = searchsortedlast(tpoints, t_active)
    return idx + 1
end

find_index!(d::AbstractDict, t, st) = get!(d, t) do
    find_index!(nothing, t, st)
end

@non_differentiable find_index!(cache, t, st)

function (pc::PiecewiseParameter)(t::T, ps, st) where {T <: Number}
    idx = find_index!(st.cache, t, st)
    return ps[idx], merge(st, (; current_index = idx))
end

number_of_shooting_constraints(pc::PiecewiseParameter) = size(pc.injected, 1)

function shooting_constraints(pc::PiecewiseParameter, ps, st)
    (; injected) = pc
    return ps[injected] .- ps[injected .- 1]
end

function shooting_constraints!(res::AbstractArray, pc::PiecewiseParameter, ps, st)
    (; injected) = pc
    return res .= ps[injected] .- ps[injected .- 1]
end

function get_timepoints(pc::PiecewiseParameter, ps, st)
    return pc.tpoints
end
