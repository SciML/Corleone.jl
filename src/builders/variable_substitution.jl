function collect_integrals!(substitutions, ex, t, gridpoints)
    if iscall(ex)
        op, args = operation(ex), arguments(ex)
        ex = op(map(args) do arg
            collect_integrals!(substitutions, arg, t, gridpoints)
        end...)
        if isa(op, Symbolics.Integral)
            if any(Base.Fix1(isequal, args), keys(substitutions))
                return substitutions[args]
            end
            sym = Symbol(:𝕃, Symbol(Char(0x2080 + length(substitutions) + 1)))
            var = Symbolics.unwrap(only(@variables ($sym)(t) = 0.0, [costvariable = true]))
            substitutions[args] = (var, op)
            # We replace with the bounds here
            lo, hi = op.domain.domain.left, op.domain.domain.right
            right = [ti for ti in gridpoints if lo <= ti <= hi]
            push!(right, hi)
            unique!(right)
            left = [ti for ti in gridpoints if ti <= lo]
            push!(left, lo)
            unique!(left)
            return sum(operation(var).(right)) - sum(operation(var).(left))
        end
    end
    return ex
end

function expand_lagrange!(builder::AbstractBuilder)
    (; system, substitutions, grids) = builder
    gridpoints = only(grids).timepoints
    t = ModelingToolkit.get_iv(system)
    costs = ModelingToolkit.get_costs(system)
    constraints = ModelingToolkit.constraints(system)
    if __process_costs(builder)
        new_costs = map(costs) do eq
            collect_integrals!(substitutions, eq, t, gridpoints)
        end
    else
        new_costs = costs
    end
    if __process_constraints(builder)
        new_constraints = map(constraints) do con
            con = Symbolics.canonical_form(con)
            new_lhs = collect_integrals!(substitutions, con.lhs, t, gridpoints)
            isa(con, Inequality) ? new_lhs ≲ con.rhs : new_lhs ~ 0
        end
    else
        new_constraints = constraints
    end
    if !isempty(substitutions)
        system = ModelingToolkit.add_accumulations(system,
            [v[1] => only(k) for (k, v) in substitutions]
        )
    end
    sys = @set system.costs = new_costs
    sys = @set sys.constraints = new_constraints
    return @set builder.system = sys
end

function append_shooting_constraints!(builder::AbstractBuilder)
    (; system, grids) = builder
    gridpoints = only(grids).timepoints
    isempty(gridpoints) && return prob
    vars = ModelingToolkit.unknowns(system)
    constraints = ModelingToolkit.constraints(system)
    for v in vars
        is_costvariable(v) && continue
        var = operation(Symbolics.unwrap(v))
        if isinput(v)
            pp = _maybecollect(ModelingToolkit.getvar(system, Symbol(var, :ᵢ), namespace=false))
            !all(is_differentialcontrol.(pp)) && continue
            ps = _maybecollect(ModelingToolkit.getvar(system, Symbol(var, :ₛ), namespace=false))
        else
            ps = _maybecollect(ModelingToolkit.getvar(system, Symbol(var, :ₛ), namespace=false))
        end
        for i in 2:size(ps, 1)
            push!(constraints,
                ps[i] - var(gridpoints[i]) ~ 0
            )
        end
    end
    return @set builder.system = system
end

function replace_shooting_variables!(builder::AbstractBuilder)
    (; system, substitutions) = builder
    # We assume first one is t0
    tshoot = get_shootingpoints(system)
    length(tshoot) <= 1 && return builder
    t = ModelingToolkit.get_iv(system)
    vars = ModelingToolkit.unknowns(system)
    empty!(substitutions)
    for v in filter(is_statevar, vars)
        ps, _ = find_shooting_pairs(system, v)
        var = operation(Symbolics.unwrap(v))
        ps = _maybecollect(ModelingToolkit.getvar(system, Symbol(var, :ₛ), namespace=false))
        for i in eachindex(tshoot)
            substitutions[var(tshoot[i])] = ps[i]
        end
    end
    new_costs = map(ModelingToolkit.get_costs(system)) do eq
        substitute(eq, substitutions)
    end
    new_constraints = map(ModelingToolkit.constraints(system)) do eq
        new_lhs = substitute(eq.lhs, substitutions)
        isa(eq, Equation) ? new_lhs ~ 0 : new_lhs ≲ 0
    end

    empty!(ModelingToolkit.constraints(system))
    append!(ModelingToolkit.constraints(system), collect(Union{Equation,Inequality}, new_constraints))
    empty!(ModelingToolkit.get_costs(system))
    append!(ModelingToolkit.get_costs(system), new_costs)
    return builder
end

function collect_explicit_timepoints!(subs, ex, vars, iv)
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
                collect_explicit_timepoints!(subs, arg, vars, iv)
            end...)
            return newex
        end
    end
    return ex
end

function create_cost_substitutions(subs, vars)
    timepoints = first.(values(subs))
    sort!(timepoints)
    unique!(timepoints)
    statevars = @variables $(gensym(:X))[1:length(vars), 1:length(timepoints)]
    newsubs = Dict()
    for (k, v) in subs
        opidx = findfirst(Base.Fix1(Base.isequal, operation(k)), vars)
        newsubs[last(v)] = getindex(statevars[1], opidx, findfirst(==(first(v)), timepoints))
    end
    return statevars, newsubs, timepoints
end
