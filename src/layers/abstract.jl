@generated function nested_eval(f, x, ps, st::NamedTuple{NAMES}) where {NAMES}
    rets = [gensym() for _ in NAMES]
    expr = Expr[]
    for (r, n) in zip(rets, NAMES)
        push!(expr, :($(r) = f(x.$(n), ps.$(n), st.$(n))))
    end
    push!(expr, :(return NamedTuple{NAMES}(($(rets...),))))
    return Expr(:block, expr...)
end

"""
$(FUNCTIONNAME)(layer, ps, st)

Total number of piecewise-continuity constraints contributed by `layer` (i.e. the
number of injected breakpoints across all `PiecewiseParameter` children).
"""
number_of_shooting_constraints(x::LuxCore.AbstractLuxLayer, ps, st) = 0
number_of_shooting_constraints(x::LuxCore.AbstractLuxWrapperLayer{T}, ps, st) where {T} = number_of_shooting_constraints(getfield(x, only(T)), ps, st)
function number_of_shooting_constraints(x::LuxCore.AbstractLuxContainerLayer{T}, ps, st) where {T}
    return sum(T) do ti
        getter = Base.Fix2(getfield, ti)
        nested_eval(number_of_shooting_constraints, getter(x), getter(ps), getter(st))
    end
end
"""
$(FUNCTIONNAME)(layer, ps, st)

Evaluate the continuity constraint contributed by `layer`.
"""
shooting_constraints(::LuxCore.AbstractLuxLayer, ps, st) = nothing
shooting_constraints(x::LuxCore.AbstractLuxWrapperLayer{T}, ps, st) where {T} = shooting_constraints(getfield(x, only(T)), ps, st)
shooting_constraints(x::LuxCore.AbstractLuxContainerLayer{T}, ps, st) where {T} = reduce(
    vcat, map(T) do ti
        getter = Base.Fix2(getfield, ti)
        nested_eval(shooting_constraints, getter(x), getter(ps), getter(st))
    end
)

shooting_constraints!(res, ::LuxCore.AbstractLuxLayer, ps, st) = res
shooting_constraints!(res, x::LuxCore.AbstractLuxWrapperLayer{T}, ps, st) where {T} = shooting_constraints!(res, getfield(x, only(T)), ps, st)
function shooting_constraints!(res, x::LuxCore.AbstractLuxContainerLayer{T}, ps, st) where {T}
    n = 0:0
    for ti in T
        getter = Base.Fix2(getfield, ti)
        layer = getter(x)
        n += 1:number_of_shooting_constraints(layer, getter(ps), getter(st))
        nested_eval(Base.Fix1(shooting_constraints!, res[n]), layer, getter(ps), getter(st))
    end
    return
end

get_lower_bound(::T, val = -Inf) where {T <: Number} = T(val)
get_upper_bound(::T, val = Inf) where {T <: Number} = T(val)

for T in (NamedTuple, AbstractArray, Base.AbstractVecOrTuple)
    @eval get_lower_bound(x::$(T), val = -Inf) = map(Base.Fix2(get_lower_bound, val), x)
    @eval get_upper_bound(x::$(T), val = Inf) = map(Base.Fix2(get_upper_bound, val), x)
end

get_lower_bound(::LuxCore.AbstractLuxLayer, ps, st) = get_lower_bound(ps)
get_lower_bound(x::LuxCore.AbstractLuxWrapperLayer{T}, ps, st) where {T} = get_lower_bound(getfield(x, only(T)), ps, st)
function get_lower_bound(x::LuxCore.AbstractLuxContainerLayer{T}, ps, st) where {T}
    return map(T) do name
        getter = Base.Fix2(getfield, name)
        nested_eval(get_lower_bound, getter(x), getter(ps), getter(st))
    end
end

get_upper_bound(::LuxCore.AbstractLuxLayer, ps, st) = get_upper_bound(ps)
get_upper_bound(x::LuxCore.AbstractLuxWrapperLayer{T}, ps, st) where {T} = get_upper_bound(getfield(x, only(T)), ps, st)
function get_upper_bound(x::LuxCore.AbstractLuxContainerLayer{T}, ps, st) where {T}
    return map(T) do name
        getter = Base.Fix2(getfield, name)
        nested_eval(get_upper_bound, getter(x), getter(ps), getter(st))
    end
end

get_bounds(x::LuxCore.AbstractLuxLayer, ps, st) = (
    get_lower_bound(x, ps, st), get_upper_bound(x, ps, st),
)

function collect_activity_pattern(timepoints::AbstractVector, x::LuxCore.AbstractLuxLayer, ps, st)
    return ones(Bool, length(timepoints), LuxCore.parameterlength(x))
end

function collect_activity_pattern(timepoints::AbstractVector, x::LuxCore.AbstractLuxWrapperLayer{T}, ps, st) where {T}
    return collect_activity_pattern(timepoints, getfield(x, only(T)), ps, st)
end

function collect_activity_pattern(timeponts::AbstractVector, x::LuxCore.AbstractLuxContainerLayer{T}, ps, st) where {T}
    return map(T) do ti
        getter = Base.Fix2(getfield, ti)
        ti => nested_eval(Base.Fix1(collect_activity_pattern, timeponts), getter(x), getter(ps), getter(st))
    end |> NamedTuple
end
