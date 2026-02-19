function Corleone.CorleoneDynamicOptProblem(
        sys::ModelingToolkit.System,
        inits::AbstractVector,
        controls::Pair...;
        algorithm::Corleone.SciMLBase.AbstractDEAlgorithm,
        shooting::Union{<:AbstractVector{<:Real}, Nothing} = nothing,
        tspan::Union{Tuple{Real, Real}, Nothing} = nothing,
        sensealg = ModelingToolkit.SciMLBase.NoAD(),
        kwargs...
    )
    iv = ModelingToolkit.get_iv(sys)
    cost = ModelingToolkit.get_costs(sys)
    constraints = ModelingToolkit.get_constraints(sys)

    lagranges = Dict()
    tpoints = Dict()
    tcollector = Base.Fix1(collect_timepoints!, tpoints)

    newcosts = map(cost) do c
        collect_integrals!(lagranges, c, iv) |> tcollector
    end

    newcons = map(constraints) do c
        c = Symbolics.canonical_form(c)
        new_con = collect_integrals!(lagranges, c.lhs, iv) |> tcollector
        isa(c, Inequality) ? new_con ≲ c.rhs : new_con ~ c.rhs
    end

    # Collect tspan
    saveats = reduce(vcat, values(tpoints))
    !isnothing(tspan) && append!(saveats, collect(tspan))
    !isnothing(shooting) && append!(saveats, collect(shooting))
    foreach(controls) do (_, grid)
        append!(saveats, collect(grid))
    end
    unique!(sort!(saveats))
    tspan = extrema(saveats)
    if !isempty(lagranges)
        # Add lagrangians and build the ODE Problem
        sys = ModelingToolkit.add_accumulations(
            sys, [v => only(arguments(k)) for (k, v) in lagranges]
        )
    end

    inputs = collect(first.(controls))
    sys = mtkcompile(sys; inputs)
    prob = ODEProblem(sys, inits, tspan, saveat = saveats, check_compatibility = false, sensealg = sensealg)

    # Get the indices of the tunables
    controls = map(controls) do (ui, tis)
        ui = Symbolics.unwrap(ui)
        i = SymbolicIndexingInterface.parameter_index(sys, ui).idx
        lo, hi = ModelingToolkit.getbounds(ui)
        u0 = Symbolics.getdefaultval(ui)
        i => ControlParameter(tis, name = Symbolics.tosymbol(operation(ui)), bounds = (lo, hi), controls = fill(u0, size(tis)))
    end

    vars = unknowns(sys)
    sort!(vars, by = Base.Fix1(SymbolicIndexingInterface.variable_index, sys))
    tunable_ic = findall(i -> ModelingToolkit.istunable(vars[i]), eachindex(vars))
    bounds_ic = map(ModelingToolkit.getbounds, vars)
    bounds_ic = (first.(bounds_ic), last.(bounds_ic))
    p_tunable = tunable_parameters(sys)
    bounds_p = map(i -> ModelingToolkit.getbounds(p_tunable[i]), filter(∉(first.(controls)), eachindex(p_tunable)))
    bounds_p = map((first.(bounds_p), last.(bounds_p)))  do bound 
        collect(Iterators.flatten(bound))
    end


    layer = if isnothing(shooting)
        SingleShootingLayer(
            prob, algorithm;
            controls,
            tunable_ic,
            bounds_ic,
            bounds_p
        )
    else
        MultipleShootingLayer(
            prob, algorithm, shooting...;
            controls,
            tunable_ic,
            bounds_ic,
            bounds_p
        )
    end
    # Get the state
    st = LuxCore.initialstates(Random.default_rng(), layer)
    symcache = isnothing(shooting) ? st.symcache : first(st).symcache
    # Build the getters
    p = tunable_parameters(sys)
    xs = map(x -> x(iv), collect(keys(tpoints)))
    symgetters = map(vcat(xs, p)) do k
        get_ = SymbolicIndexingInterface.getsym(symcache, k)
        sym_ = Symbolics.tosymbol(maybeop(k))
        sym_, get_
    end
    getters = Tuple(last.(symgetters))
    # Build the substitution for the costs & constraints
    vars_subs = Dict()
    foreach(keys(tpoints)) do k
        tvals = unique(sort!(tpoints[k]))
        foreach(tvals) do ti
            get!(vars_subs, k(ti), Expr(:call, getindex, k.name, findfirst(ti .== saveats)))
        end
    end

    # Arguments for the objective and constraints
    args_ = first.(symgetters)

    # We currently assume a single cost
    costbody = SymbolicUtils.Code.toexpr(substitute(only(newcosts), vars_subs))

    costfn = :(($(args_...),) -> $costbody)

    costfun = @RuntimeGeneratedFunction(
        costfn
    )

    costs = let predictor = layer, obs = getters, objective = costfun
        (ps, st) -> begin
            traj, _ = predictor(nothing, ps, st)
            vars = map(obs) do getter
                getter(traj)
            end
            objective(vars...)
        end
    end

    if !isempty(newcons)
        res = gensym(:con)
        conbody = map(enumerate(newcons)) do (i, con)
            ex = SymbolicUtils.Code.toexpr(substitute(con.lhs, vars_subs))
            :($(res)[$(i)] = $ex)
        end
        push!(conbody, :(return $(res)))

        confn = :(($res, $(args_...)) -> $(conbody...))

        confun = @RuntimeGeneratedFunction(
            confn
        )
        lcons = -Inf .* map(Base.Fix2(isa, Inequality), newcons)
        n_cons = size(newcons, 1)
        ucons = zeros(Float64, n_cons)
        n_shoot = Corleone.get_number_of_shooting_constraints(layer)
        append!(lcons, zeros(n_shoot))
        append!(ucons, zeros(n_shoot))

        cons = let predictor = layer, obs = getters, constr = confun, ncon = n_cons
            (res, ps, st) -> begin
                traj, _ = predictor(nothing, ps, st)
                vars = map(obs) do getter
                    getter(traj)
                end
                @views constr(res[1:ncon], vars...)
                @views Corleone.shooting_constraints!(res[(ncon + 1):end], traj)
                return res
            end
        end
    elseif isa(layer, MultipleShootingLayer)
        n_shoot = Corleone.get_number_of_shooting_constraints(layer)
        lcons = ucons = zeros(n_shoot)
        cons = let layer = layer
            (res, ps, st) -> begin
                traj, _ = layer(nothing, ps, st)
                @views Corleone.shooting_constraints!(res, traj)
                return res
            end
        end
    else
        ucons = lcons = cons = nothing
    end

    return CorleoneDynamicOptProblem{typeof(layer), typeof(getters), typeof(costs), typeof(cons), typeof(lcons)}(
        layer,
        getters,
        costs,
        cons,
        lcons,
        ucons
    )
end
