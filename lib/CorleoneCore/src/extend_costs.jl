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
            var = Symbolics.unwrap(only(@variables ($sym)(t) = 0.0, [costvariable = true]))
            subs[ex] = var
            return var
        end
    end
    return ex
end

function extend_costs(sys)
    costs = ModelingToolkit.get_costs(sys)
    consol = ModelingToolkit.get_consolidate(sys)
    t = ModelingToolkit.get_iv(sys)
    subs = Dict()
    new_costs = Num[]
    for ex in costs
        push!(new_costs, collect_integrals!(subs, Symbolics.unwrap(ex), t) |> Num)
    end
    inv_subs = Dict([v => k for (k, v) in subs])
    # TODO Maybe use MTKs add_accumulation here.
    # Main point not to do this, we might have nested / multiple arguments here
    D = Differential(t)
    new_eqs = [
        D(k) ~ substitute(arguments(v)[1], subs) for (k, v) in inv_subs
    ]
    new_vars = collect(keys(inv_subs))
    lag_sys = ODESystem(
        new_eqs, t, new_vars, [], name=nameof(sys),
        # costs = new_costs,
    )

    newsys = extend(lag_sys, sys)
    newsys = @set newsys.costs = new_costs
    newsys = @set newsys.consolidate = consol
    newsys = @set newsys.tspan = ModelingToolkit.get_tspan(sys)
end

# We have a very specific case here.
# If the objective is of the form min sum(args...)
# and any of the args is equal to the iv of the sys, then
# we transform the time
# If the objective is purely min iv(sys) also
function change_of_variables(sys)
    costs = ModelingToolkit.get_costs(sys)
    consol = ModelingToolkit.get_consolidate(sys)
    iv = independent_variable(sys)
    transform_iv = false
    for obj in costs
        if !istree(obj)
            transform_iv = isequal(obj, iv)
        else
            transform_iv = (operation(obj) ∈ (+, sum)) && any(Base.Fix1(isequal, iv), arguments(obj))
        end
    end
    transform_iv || return sys
    D = Differential(iv)
    varsym = Symbol(:τ, :ᵢ, :ᵥ)
    psym = Symbol(:T, :ᵢ, :ᵥ)
    p = @parameters ($psym) = 1.0 [bounds = (eps(), Inf)]
    x = @variables ($varsym)(iv) = 0.0
    Dx = Differential(x[1])
    # TODO Double check here!
    eqs = Equation[
        D(x[1])~p[1],
        iv~inv(p[1])*x[1]
    ]
    paramsys = ODESystem(Equation[], iv, x, p, name=nameof(sys))
    new_costs = map(costs) do expr
        if !istree(expr) && isequal(expr, iv)
            return x[1]
        elseif istree(expr) && (operation(expr) ∈ (+, sum)) && any(Base.Fix1(isequal, iv), arguments(expr))
            return expr - iv + x[1]
        end
        expr
    end
    newsys = extend(paramsys, sys)
    newsys = change_independent_variable(newsys, x[1], eqs, add_old_diff=false, simplify=true, fold=true)
    #newsys = @set newsys.costs = costs
    #newsys = @set newsys.consolidate = consol
    newsys
end

function find_explicit_timepoints!(subs, ex, vars, iv)
    if iscall(ex)
        op, args = operation(ex), arguments(ex)
        if op ∈ vars && !Base.isequal(args[1], iv)
            if any(Base.Fix1(isequal, ex), keys(subs))
                return last(subs[ex]) 
            end
            sym = gensym(Symbol(op)) #, Symbol(Char(0x2080 + length(subs) + 1)))
            var = Symbolics.unwrap(only(@variables ($sym)))
            subs[ex] = (only(args), var)
            return var
        else 
            newex = op(map(args) do arg
                find_explicit_timepoints!(subs, arg, vars, iv)
            end...)
            return newex 
        end
    end
    return ex
end

struct SystemEvaluator{OOP, IIP, T} <: Function 
    "Out-of-place function"
    f_oop::OOP 
    "In-place function"
    f_iip::IIP 
    "Evaluation timepoints"
    saveat::T 
end

function _get_normalized_constraints(sys)
    csys = ModelingToolkit.get_constraintsystem(sys) 
    isnothing(csys) && return Num[] 
    eqs = map(x->x.lhs, Symbolics.canonical_form.(equations(csys)))  
end 

function build_evaluators(sys)
    iv = ModelingToolkit.get_iv(sys)
    varset = operation.(ModelingToolkit.unknowns(sys))
    costs = ModelingToolkit.get_costs(sys) 
    idx = [!is_costvariable(xi) for xi in costs]
    objective
    cons = _get_normalized_constraints(sys)
    
end

# Right now no controls etc are supported here. We assume just states and parameters
function build_evaluator(sys, expressions, vars, iv)
    config = expand_mayer_terms(expressions, vars, iv)
    expressions = build_function(config.expressions, vars, parameters)
end

function expand_mayer_terms(expressions, vars, iv)
    subs = Dict()
    new_expressions = map(expressions) do ex 
        find_explicit_timepoints!(subs, Symbolics.unwrap(ex), vars, iv)
    end
    tpoints = unique!(sort!(first.(values(subs))))
    for (k, v) in subs
        opidx = findfirst(Base.Fix1(Base.isequal, operation(k)), varset)
        subs[k] = ((opidx, findfirst(==(v[1]), tpoints)), v[2])  
    end
    config = (; expressions = new_expressions, timepoints = tpoints, substitutions = subs)
end
