function collect_integrals!(subs, ex, t)
    if iscall(ex)
        op, args = operation(ex), arguments(ex)
        ex = op(map(args) do arg
            collect_integrals!(subs, arg, t)
        end...)

        if isa(op, Integral)
            if any(Base.Fix1(isequal, ex), keys(subs))
                return subs[ex]
            end
            sym = Symbol(:𝕃, Symbol(Char(0x2080 + length(subs) + 1)))
            var = Symbolics.unwrap(only(@variables ($sym)(t) = 0.0))
            subs[ex] = var
            return var
        end
    end
    return ex
end

function extend_integrals(sys, expressions...)
    expressions = Symbolics.unwrap.(expressions)
    t = independent_variable(sys)
    subs = Dict()
    expressions = map(expressions) do ex
        collect_integrals!(subs, ex, t)
    end
    inv_subs = Dict([v => k for (k, v) in subs])
    D = Differential(t)
    new_eqs = [
        D(k) ~ substitute(arguments(v)[1], subs) for (k, v) in inv_subs
    ]
    new_vars = collect(keys(inv_subs))
    lag_sys = ODESystem(
        new_eqs, t, new_vars, [], name=Symbol(:Lagrange, :_, nameof(sys))
    )
    extend(sys, lag_sys), expressions
end

function collect_forall!(subs, ex)
    if iscall(ex)
        op, args = operation(ex), arguments(ex)
        ex = op(map(args) do arg
            collect_forall!(subs, arg)
        end...)

        if isa(op, CorleoneCore.ForAll)
            arg = only(arguments(ex))
            i, sub_ = get!(subs, arg) do
                (length(subs) + 1, Dict())
            end
            sym = get!(sub_, op.timepoint) do
                j = length(sub_) + 1
                sym = Symbol(:𝕍, Symbol(Char(0x2080 + i)), Symbol(Char(0x2080 + j)))
                only(@variables ($sym))
            end
            return sym
        end
    end
    return ex
end

function _prepare_expression(x)
    Symbolics.unwrap(x), (-Inf, Inf)
end

function _prepare_expression(x::Union{Equation,Inequality})
    Symbolics.unwrap(Symbolics.canonical_form(x).lhs), isa(x, Equation) ? (0.0, 0.0) : (-Inf, 0.0)
end

function extend_forall(sys, expressions...)
    rets = _prepare_expression.(expressions)
    expressions = first.(rets)
    bounds = last.(rets)
    subs = Dict()
    expressions = map(expressions) do ex
        collect_forall!(subs, ex)
    end

    @info subs
    # Build the observed function
    args = []
    obs = []
    for (k, (_, v)) in subs
        ex = Num(k)
        saveats = collect(keys(v))
        vars = collect(values(v))
        idx = sortperm(saveats)
        permute!(saveats, idx)
        permute!(vars, idx)
        push!(args, vars)
        specs = (;
            expression=ex,
            saveats=saveats,
        )
        push!(obs, specs)
    end

    cons = reduce(vcat, map(enumerate(expressions)) do (i, expr)
        isequal(bounds[i]...) ? expr ~ 0 : expr ≲ 0
    end)

    consys = ConstraintsSystem(
        cons,
        reduce(vcat, args),
        parameters(sys),
        name=Symbol(:DynamicConstraints, :_, nameof(sys))
    )
    (foop, fiip), lb, ub = generate_function(consys, expression=Val{false})

    #foop, fiip = build_function(
    #    reduce(vcat, expressions),
    #   reduce(vcat, args),
    #   ModelingToolkit.reorder_parameters(sys, parameters(sys))...;
    #   expression = Val{false}
    #)
    TestFun(tuple(obs...), foop, fiip, bounds)
end
