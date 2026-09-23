"""
"""
@concrete terse struct DynamicFunctionLayer <: LuxCore.AbstractLuxLayer
    "Additional parameters"
    parameter
    "State constructor"
    state
    "Out of place function"
    foop 
    "(Optional) Is in place function"
    fiip
end

function DynamicFunctionLayer(f::Base.Callable; 
        isinplace = nothing,
        parameters = (rng) -> (;),
        state = (rng) -> (;),  
    )
    DynamicFunctionLayer(parameters, state, f, isinplace)
end

LuxCore.initialparameters(rng::Random.AbstractRNG, layer::DynamicFunctionLayer) = layer.parameter(rng)
LuxCore.initialstates(rng::Random.AbstractRNG, layer::DynamicFunctionLayer) = layer.state(rng)

function (layer::DynamicFunctionLayer)(x::Trajectory, ps, st::NamedTuple)
    out, st = layer.foop(x, ps, st)
    out, st
end

function (layer::DynamicFunctionLayer)(res, x::Trajectory, ps, st::NamedTuple)
    st = layer.fiip(res, x, ps, st)
    res, st
end

function (layer::DynamicFunctionLayer{<:Any, <:Any, <:Any, Nothing})(res, x::Trajectory, ps, st::NamedTuple)
    _res, st = layer(x, ps, st)
    res .= _res
    res, st
end

process_expression(x::Symbol) = x, -Inf, Inf

function process_expression(expr::Expr) 
    op, args... = expr.args 
    @info op args
    if op == :(==)
        return Expr(:call, -, args...), 0., 0.
    elseif op == :(<=)
        return Expr(:call, -, args...), -Inf, 0.
    elseif op == :(>=)
        return Expr(:call, -, args...), -Inf, 0.0
    else
        return expr, -Inf, Inf
    end
end

function process_expression(exprs::Base.AbstractVecOrTuple)
    lb = zeros(length(exprs))
    ub = zeros(length(exprs)) 
    cons = map(enumerate(exprs)) do (i,expr)
        con, lb_, ub_ = process_expression(expr)
        lb[i] = lb_
        ub[i] = ub_
        con
    end
    cons, lb, ub
end

function DynamicFunctionLayer(layer::ShootingLayer, expr...; 
    rng::Random.AbstractRNG = Random.default_rng(),
    eval_expression::Val{E} = Val{true}(),
    kwargs...) where E
    timepoints = collect_timegrid(layer, LuxCore.setup(rng, layer)...)
    T = eltype(timepoints)
    @info expr
    expr, lb, ub = process_expression(expr)
    parser = Parser.Parser{T}(layer.sys)
    foop, fiip = parser(expr...; timepoints)
    foop, fiip = if E 
        eval(foop), eval(fiip) 
    else
        throw(ArgumentError("Only eval_expression = Val{true}() is implemented."))
    end
    saveats = vcat(timepoints, (collect ∘ keys)(parser.indexgrid))
    unique!(sort!(saveats))
    state_constructor = let saveat = saveats, lb = lb, ub = ub 
        (rng) -> (; saveat = saveat, lb = lb, ub = ub)
    end
    DynamicFunctionLayer(
        foop, 
        state = state_constructor, 
        isinplace = fiip
    )
end