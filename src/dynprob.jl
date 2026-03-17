"""
$(TYPEDEF)

A layer that wraps a shooting layer to solve dynamic optimization problems with an objective function and constraints.

# Fields
$(FIELDS)

# Description

The `DynamicOptimizationLayer` combines a shooting layer (single or multiple) with an objective function and optional constraints
to formulate a complete dynamic optimization problem. The objective and constraints are specified as symbolic expressions that
are evaluated on the computed trajectory.
"""
struct DynamicOptimizationLayer{N, L, G, O, C, CB} <: LuxCore.AbstractLuxWrapperLayer{:layer}
    "The name of the layer, used for display and logging purposes."
    name::N
    "The wrapped shooting layer used to produce trajectories."
    layer::L
    "The getter for all symbols"
    getters::G
    "The objective function"
    objective::O
    "The constraints function"
    constraints::C
    "The lower bounds for the constraints"
    lcons::CB
    "The upper bounds for the constraints"
    ucons::CB
end

_extract_timepoints(x::Number) = [x]
_extract_timepoints(x::Expr) = begin
    @assert x.head == :vect "Timepoints must be provided as a scalar or vector, e.g. `x(1.0)` or `x([1.0, 2.0])"
    reduce(vcat, x.args)
end

_collect_timepoints!(::Dict{Symbol, <:AbstractVector}, ::Any) = nothing

function _collect_timepoints!(collector::Dict{Symbol, <:AbstractVector}, ex::Expr)
    if ex.head == :call
        if ex.args[1] ∈ keys(collector)
            append!(collector[ex.args[1]], _extract_timepoints(ex.args[2]))
        else
            for arg in ex.args
                _collect_timepoints!(collector, arg)
            end
        end
    end
    return
end

_extract_timeindex(x::Number, indices) = indices[x]
_extract_timeindex(x::Expr, indices) = begin
    @assert x.head == :vect "Timepoints must be provided as a scalar or vector, e.g. `x(1.0)` or `x([1.0, 2.0])"
    reduce(vcat, map(Base.Fix2(_extract_timeindex, indices), x.args))
end

replace_timepoints(x::Any, replacer) = x

function replace_timepoints(x::Expr, replacer::Dict{Symbol, <:Dict})
    if x.head == :call
        if x.args[1] ∈ keys(replacer)
            return Expr(
                :call, :getindex, x.args[1],
                _extract_timeindex(x.args[2], replacer[x.args[1]])
            )
        else
            return Expr(x.head, map(arg -> replace_timepoints(arg, replacer), x.args)...)
        end
    end
    return
end

function find_indices(points, grid)
    t0, tinf = extrema(grid)
    return Dict(
        map(points) do p
            p <= t0 && return p => firstindex(grid)
            p >= tinf && return p => lastindex(grid)
            p => searchsortedlast(grid, p)
        end...
    )
end

function build_iip(problem, header::AbstractVector{<:Expr}, exprssions::AbstractVector{<:Expr}, offset::Int64 = 0)
    returns = gensym()
    exprs = [:($(returns)[$(i + offset)] = $(exprssions[i])) for i in eachindex(exprssions)]
    push!(exprs, :(return $returns))
    headercall = Expr(:call, gensym(), returns, :trajectory)
    oop_expr = Expr(:function, headercall, Expr(:block, header..., exprs...))
    return observed = @RuntimeGeneratedFunction(oop_expr)
end

function build_oop(problem, header::AbstractVector{<:Expr}, expressions::AbstractVector{<:Expr})
    returns = [gensym() for _ in expressions]
    exprs = [:($(returns[i]) = $(expressions[i])) for i in eachindex(returns)]
    if length(expressions) > 1
        push!(exprs, :(return [$(returns...)]))
    else
        push!(exprs, :(return $(returns[1])))
    end
    headercall = Expr(:call, gensym(), :trajectory)
    oop_expr = Expr(:function, headercall, Expr(:block, header..., exprs...))
    return observed = @RuntimeGeneratedFunction(oop_expr)
end

function normalize_constraint(expr::Expr, ::Type{T}) where {T}
    @assert expr.head == :call "The expression is not a call."
    op, a, b = expr.args
    return if op == :(<=)
        Expr(:call, :(-), a, b), T(-Inf), zero(T)
    elseif op == :(>=)
        Expr(:call, :(-), b, a), T(-Inf), zero(T)
    elseif op == :(==)
        Expr(:call, :(-), a, b), zero(T), zero(T)
    else
        throw(error("The operator $(op) is not suppported to define constraints. Only ==, <=, and >= are supported."))
    end
end

function DynamicOptimizationLayer(layer::LuxCore.AbstractLuxLayer, objective::Expr, constraints::Expr...; name = gensym(:observed))
    problem = Corleone.get_problem(layer)
    T = eltype(problem.u0)
    lb = fill(zero(T), length(constraints) + get_number_of_shooting_constraints(layer))
    ub = fill(zero(T), length(constraints) + get_number_of_shooting_constraints(layer))

    constraints = map(enumerate(constraints)) do (i, con)
        con, lb[i], ub[i] = normalize_constraint(con, T)
        con
    end
    expressions = vcat(objective, constraints...)
    symbols = vcat(variable_symbols(problem), parameter_symbols(problem))
    tspan = get_tspan(layer)
    collector = Dict([vi => eltype(tspan)[] for vi in symbols])
    foreach(expressions) do ex
        _collect_timepoints!(collector, ex)
    end
    # Find the indices
    timegrid = Corleone.get_timegrid(layer)
    foreach(values(collector)) do tps
        append!(timegrid, tps)
    end
    unique!(sort!(timegrid))
    layer = remake(layer, saveat = timegrid)
    replacer = Dict([ki => find_indices(vi, timegrid) for (ki, vi) in zip(keys(collector), values(collector)) if !isempty(vi)])
    new_exprs = map(expressions) do ex
        replace_timepoints(ex, replacer)
    end
    header = map(collect(keys(replacer))) do k
        if is_parameter(problem, k)
            return :($(k) = trajectory.ps[$(QuoteNode(k))])
        else
            return :($(k) = trajectory[$(QuoteNode(k))])
        end
    end
    getter = nothing
    objective = build_oop(problem, header, new_exprs[1:1])
    constraints = build_iip(problem, header, new_exprs[2:end], get_number_of_shooting_constraints(layer))
    return DynamicOptimizationLayer{typeof(name), typeof(layer), typeof(getter), typeof(objective), typeof(constraints), typeof(lb)}(
        name, layer, getter,
        objective, constraints, lb, ub
    )
end

function (obs::DynamicOptimizationLayer)(x::Nothing, ps, st)
    trajectory, st = obs.layer(x, ps, st)
    obj = obs.objective(trajectory)
    return obj, st
end

function (obs::DynamicOptimizationLayer)(x, ps, st)
    trajectory, st = obs.layer(x, ps, st)
    shooting_constraints!(x, trajectory)
    obs.constraints(x, trajectory)
    return x, st
end

# A simple wrapper for reconstructing the parameters
struct WrappedFunction{F, P}
    f::F
    parameter::P
end

(f::WrappedFunction)(u, p) = first(f.f(nothing, f.parameter(u), p))
(f::WrappedFunction)(res, u, p) = first(f.f(res, f.parameter(u), p))

function WrappedFunction(::Any, f, p, st; kwargs...) end

function to_vec(::Any, p) end

#

"""
$(SIGNATURES)

Construct a SciML `OptimizationProblem` from a [`CorleoneDynamicOptProblem`](@ref).
"""
function SciMLBase.OptimizationProblem(
        prob::DynamicOptimizationLayer, ad::SciMLBase.ADTypes.AbstractADType,
        ps = nothing,
        st = nothing;
        rng::Random.AbstractRNG = Random.default_rng(),
        vectorizer,
        sense = nothing,
        kwargs...
    )
    ps = something(ps, LuxCore.initialparameters(rng, prob))
    st = something(st, LuxCore.initialstates(rng, prob))
    optf = SciMLBase.OptimizationFunction(prob, ad; vectorizer, rng, ps, st, kwargs...)
    u0, lb, ub = map(Base.Fix1(to_vec, vectorizer), (ps, Corleone.get_bounds(prob)...))
    return SciMLBase.OptimizationProblem(optf, u0, st; lb, ub, lcons = prob.lcons, ucons = prob.ucons, sense = sense)
end

"""
$(SIGNATURES)

Construct a SciML `OptimizationFunction` from a [`CorleoneDynamicOptProblem`](@ref).
"""
function SciMLBase.OptimizationFunction(
        prob::DynamicOptimizationLayer, ad::SciMLBase.ADTypes.AbstractADType;
        vectorizer,
        rng::Random.AbstractRNG = Random.default_rng(),
        ps = LuxCore.initialparameters(rng, prob),
        st = LuxCore.initialstates(rng, prob),
        kwargs...
    )
    wrapper = WrappedFunction(vectorizer, prob, ps, st)
    return SciMLBase.OptimizationFunction{true}(wrapper, ad; cons = wrapper, kwargs...)
end
