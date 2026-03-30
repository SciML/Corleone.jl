module CorleoneModelingToolkitExtension

@info "MTK Extension loaded!"

using Corleone
using ModelingToolkit
using SciMLBase
using SymbolicIndexingInterface

using ModelingToolkit.Symbolics
using ModelingToolkit.SymbolicUtils
using ModelingToolkit.SymbolicUtils.Code

import Corleone: SingleShootingLayer, MultipleShootingLayer, DynamicOptimizationLayer
import Corleone: ControlParameter, FixedControlParameter


function Corleone.ControlParameter(x::Union{Num, SymbolicUtils.BasicSymbolic}, tpoints::AbstractVector)
    u0 = Symbolics.getdefaultval(x)
    lb, ub = ModelingToolkit.getbounds(x)
    @info lb
    return ControlParameter(
        tpoints,
        name = x,
        controls = (rng, t) -> fill(u0, size(t, 1)),
        bounds = (t) -> (fill(lb, size(t, 1)), fill(ub, size(t, 1)))
    )
end

Corleone.remake_system(sys::ModelingToolkit.AbstractSystem, args...) = sys

function Corleone.SingleShootingLayer(
        sys::ModelingToolkit.AbstractSystem,
        defaults,
        controls...;
        algorithm::SciMLBase.AbstractDEAlgorithm,
        tspan::Tuple,
        saveat = [],
        quadrature_indices = [],
        kwargs...
    )
    input_vars = ModelingToolkit.inputs(sys)
    @assert isempty(setdiff(input_vars, first.(controls))) "Not all inputs of the system are present in the control specs"

    ttype = promote_type(typeof.(tspan)...)
    sys = mtkcompile(sys, inputs = input_vars, sort_eqs = true)
    quadrature_indices = [isa(id, Int) ? id : variable_index(sys, id) for id in quadrature_indices]
    ps = tunable_parameters(sys)
    saveats = reduce(vcat, collect.(last.(controls)))
    append!(saveats, ttype.(saveat))
    params = map(filter(!isinitial, ps)) do var
        idx = findfirst(Base.Fix1(isequal, var), first.(controls))
        isnothing(idx) && return ControlParameter(var, ttype[0])
        return ControlParameter(controls[idx]...)
    end
    tunable_ic = findall(ModelingToolkit.istunable, unknowns(sys))
    unknown_bounds = ModelingToolkit.getbounds(unknowns(sys))
    bounds_ic = let bounds = unknown_bounds
        (t0) -> (bounds.lb, bounds.ub)
    end
    # Extract sensealg for ODEProblem, but don't pass to SingleShootingLayer
    # (InitialCondition doesn't accept sensealg kwarg)
    sensealg = get(kwargs, :sensealg, nothing)
    odep_kwargs = filter!(kw -> first(kw) !== :sensealg, collect(pairs(kwargs)))
    prob = ODEProblem{true, SciMLBase.FullSpecialize()}(sys, defaults, tspan; saveat = saveats, build_initializeprob = false, sensealg, odep_kwargs...)
    return Corleone.SingleShootingLayer(prob, params...; algorithm = algorithm, tunable_ic, bounds_ic, quadrature_indices, NamedTuple(odep_kwargs)...)
end

function Corleone.MultipleShootingLayer(
        sys::ModelingToolkit.AbstractSystem,
        defaults,
        controls...;
        algorithm,
        tspan,
        shooting,
        kwargs...
    )
    single_layer = Corleone.SingleShootingLayer(sys, defaults, controls...; algorithm, tspan, kwargs...)
    return Corleone.MultipleShootingLayer(single_layer, shooting...)
end

function collect_timepoints!(tpoints, ex)
    if iscall(ex)
        op, args = operation(ex), arguments(ex)
        if SymbolicUtils.issym(op) && isa(first(args).val, Number) && length(args) == 1
            tp = first(args).val
            vars = get!(tpoints, op, typeof(tp)[])
            push!(vars, tp)
        end
        return op(
            map(args) do x
                collect_timepoints!(tpoints, x)
            end...
        )
    end
    return ex
end

Corleone._maybesymbolifyme(x::SymbolicUtils.BasicSymbolic) = iscall(x) && operation(x) != ModelingToolkit.Initial ? Symbol(operation(x)) : Symbol(x)
Corleone._maybesymbolifyme(x::Num) = Corleone._maybesymbolifyme(Symbolics.unwrap(x))

function collect_integrals!(subs, ex, t)
    if iscall(ex)
        op, args = operation(ex), arguments(ex)
        ex = op(
            map(args) do arg
                collect_integrals!(subs, arg, t)
            end...
        )
        if isa(op, Symbolics.Integral)
            var = get!(subs, ex) do
                sym = Symbol(:𝕃, Symbol(Char(0x2080 + length(subs) + 1)))
                var = Symbolics.unwrap(only(ModelingToolkit.@variables ($sym)(t) = 0.0 [tunable = false, bounds = (0.0, 0.0)])) # [costvariable = true]))
                var
            end
            lo, hi = op.domain.domain.left, op.domain.domain.right
            return operation(var)(hi) - operation(var)(lo)
        end
    end
    return ex
end

function collect_expr(ex, replacer::Dict)
    if iscall(ex)
        op, args = operation(ex), arguments(ex)
        args = map(args) do arg
            collect_expr(arg, replacer) |> toexpr
        end
        return Expr(:call, Symbol(op), args...)
    end
    var = Symbol(ex)
    return get(replacer, var, toexpr(ex))
end

maybenormalize(ex) = nothing, ex
maybenormalize(ex::Symbolics.Inequality) = begin
    x = Symbolics.canonical_form(ex)
    operation(x), x.lhs
end

function Corleone.DynamicOptimizationLayer(
        sys::ModelingToolkit.AbstractSystem,
        defaults,
        controls,
        exprs...;
        algorithm,
        tspan = nothing,
        shooting = [],
        kwargs...
    )
    iv = ModelingToolkit.get_iv(sys)
    lagranges = Dict()
    tpoints = Dict()
    tcollector = Base.Fix1(collect_timepoints!, tpoints)

    exprs = map(exprs) do expr
        newexp = collect_integrals!(lagranges, Symbolics.unwrap(expr), iv) |> tcollector

    end
    saveats = reduce(vcat, values(tpoints))
    !isnothing(tspan) && append!(saveats, collect(tspan))
    tspan = extrema(saveats)
    unique!(sort!(saveats))
    tspan = extrema(saveats)
    if !isempty(lagranges)
        # Add lagrangians and build the ODE Problem
        sys = ModelingToolkit.add_accumulations(
            sys, [v => only(arguments(k)) for (k, v) in lagranges]
        )
    end

    # Normalize controls to a vector for consistent splatting
    controls_vec = controls isa Pair ? [controls] : controls

    if isempty(shooting)
        shooting_layer = Corleone.SingleShootingLayer(
            sys, defaults, controls_vec...;
            algorithm = algorithm,
            tspan = tspan,
            quadrature_indices = values(lagranges),
            kwargs...
        )
    else
        shooting_layer = Corleone.MultipleShootingLayer(
            sys, defaults, controls_vec...;
            algorithm = algorithm,
            tspan = tspan,
            shooting = shooting,
            quadrature_indices = values(lagranges),
            kwargs...
        )
    end

    replacer = Dict{Symbol, Expr}()
    for p in tunable_parameters(sys)
        ModelingToolkit.isinitial(p) && continue
        ModelingToolkit.isinput(p) && continue
        replacer[Symbol(p)] = Expr(:call, Symbol(p), first(tspan))
    end
    exprs = map(exprs) do expr
        if isa(expr, Symbolics.Inequality) || isa(expr, Symbolics.Equation)
            expr = Symbolics.canonical_form(expr)
            new_lhs = collect_expr(expr.lhs, replacer)
            op = isa(expr, Symbolics.Inequality) ? :(<=) : :(==)
            return Expr(:call, op, new_lhs, 0)
        end
        collect_expr(expr, replacer)
    end
    return Corleone.DynamicOptimizationLayer(shooting_layer, exprs...)
end

end
