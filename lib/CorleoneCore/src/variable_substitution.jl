"""
$(TYPEDEF)

Takes in a `System` with defined `cost` and optional `constraints` and `consolidate` together with a [`ShootingGrid`](@ref) and [`AbstractControlFormulation`](@ref)s and build the related `OptimizationProblem`. 
"""
struct OCProblemBuilder{S,C,G,I}
    "The system"
    system::S
    "The controls"
    controls::C
    "The grid"
    grids::G
    "The initialization"
    initialization::I
    "Substitutions"
    substitutions::Dict
end

function Base.show(io::IO, prob::OCProblemBuilder)
    (; system) = prob
    cost = ModelingToolkit.get_consolidate(system)(ModelingToolkit.get_costs(system))
    cons = ModelingToolkit.constraints(system)
    eqs = ModelingToolkit.equations(system)
    println(io, "min $(cost)")
    println(io, "")
    println(io, "s.t.")
    println(io, "")
    println(io, "Dynamics $(nameof(system))")
    foreach(eqs) do eq
        println(io, "     ", eq)
    end
    println(io, "Observed $(nameof(system))")
    foreach(ModelingToolkit.observed(system)) do eq
        println(io, "     ", eq)
    end

    println(io, "Constraints")
    foreach(cons) do eq
        println(io, "     ", eq)
    end
end

function OCProblemBuilder(sys::ModelingToolkit.System, controls::C, grids::G, inits::I, subs::Dict) where {C<:Tuple,G<:Tuple,I<:Tuple}
    OCProblemBuilder{typeof(sys),C,G,I}(
        sys, controls, grids, inits, subs
    )
end

function OCProblemBuilder(sys::ModelingToolkit.System, args...)
    controls = filter(Base.Fix2(isa, AbstractControlFormulation), args)
    grid = filter(Base.Fix2(isa, ShootingGrid), args)
    inits = filter(Base.Fix2(isa, AbstractNodeInitialization), args)
    OCProblemBuilder{typeof(sys),typeof(controls),typeof(grid),typeof(inits)}(
        sys, controls, grid, inits, Dict()
    )
end

function (prob::OCProblemBuilder)(; kwargs...)
    # Extend the costs 
    prob = expand_lagrange!(prob)
    # Extend the controls 
    prob = @set prob.system = (only(prob.grids) ∘ tearing)(foldl(∘, prob.controls, init=identity)(prob.system))
    prob = replace_shooting_variables!(prob)
    prob = append_shooting_constraints!(prob)
    prob = @set prob.system = complete(prob.system; add_initial_parameters=false)
    return prob
end

function SciMLBase.OptimizationFunction{IIP}(prob::OCProblemBuilder, adtype::SciMLBase.ADTypes.AbstractADType, alg::DEAlgorithm, args...; kwargs...) where {IIP}
    (; system) = prob
    @assert ModelingToolkit.iscomplete(system) "The system is not complete."
    constraints = ModelingToolkit.constraints(system)
    costs = ModelingToolkit.get_costs(system) 
    consolidate = ModelingToolkit.get_consolidate(system)
    objective = OptimalControlFunction{true}(
        costs, prob, alg, args...; consolidate=consolidate, kwargs...
    )
    if !isempty(constraints)
        constraints = OptimalControlFunction{IIP}(
            map(x -> x.lhs, constraints), prob, alg, args...; kwargs...
        )
    else
        constraints = nothing
    end
    return OptimizationFunction{IIP}(
        objective, adtype, cons=constraints,
    )
end

function SciMLBase.OptimizationProblem{IIP}(prob::OCProblemBuilder, adtype::SciMLBase.ADTypes.AbstractADType, alg::DEAlgorithm, args...; kwargs...) where {IIP}
    (; system) = prob
    @assert ModelingToolkit.iscomplete(system) "The system is not complete."
    constraints = ModelingToolkit.constraints(system)
    f = OptimizationFunction{IIP}(prob, adtype, alg, args...; kwargs...)
    predictor = f.f.predictor
    u0 = get_p0(predictor)
    lb, ub = get_bounds(predictor)
    if !isempty(constraints)
        lcons = zeros(eltype(u0), size(constraints, 1))
        ucons = zeros(eltype(u0), size(constraints, 1))
        for (i, c) in enumerate(constraints)
            isa(c, Equation) && continue
            lcons[i] = eltype(u0)(-Inf)
        end
    else
        lcons = ucons = nothing
    end
    OptimizationProblem{IIP}(f, u0; lb, ub, lcons, ucons)
end

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

function expand_lagrange!(prob::OCProblemBuilder)
    (; system, substitutions, grids) = prob
    gridpoints = only(grids).timepoints
    t = ModelingToolkit.get_iv(system)
    costs = ModelingToolkit.get_costs(system)
    constraints = ModelingToolkit.constraints(system)
    new_costs = map(costs) do eq
        collect_integrals!(substitutions, eq, t, gridpoints)
    end
    new_constraints = map(constraints) do con
        con = Symbolics.canonical_form(con)
        new_lhs = collect_integrals!(substitutions, con.lhs, t, gridpoints)
        new_lhs ≲ con.rhs
    end
    system = ModelingToolkit.add_accumulations(system,
        [v[1] => only(k) for (k, v) in substitutions]
    )
    sys = @set system.costs = new_costs
    sys = @set sys.constraints = new_constraints
    return @set prob.system = sys
end

function append_shooting_constraints!(prob::OCProblemBuilder)
    (; system, grids) = prob
    gridpoints = only(grids).timepoints
    isempty(gridpoints) && return prob
    vars = ModelingToolkit.unknowns(system)
    constraints = ModelingToolkit.constraints(system)
    for v in vars
        (ModelingToolkit.isinput(v) || is_costvariable(v)) && continue
        var = operation(Symbolics.unwrap(v))
        ps = _maybecollect(ModelingToolkit.getvar(system, Symbol(var, :ₛ), namespace=false))
        for i in 2:size(ps, 1)
            push!(constraints,
                ps[i] - var(gridpoints[i]) ~ 0
            )
        end
    end
    return @set prob.system = system
end

function replace_shooting_variables!(prob::OCProblemBuilder)
    (; system, substitutions) = prob
    # We assume first one is t0 
    tshoot = get_shootingpoints(system)
    length(tshoot) <= 1 && return prob
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
    return prob
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

struct OptimalControlFunction{OOP,IIP,P,C}
    f_oop::OOP
    f_iip::IIP
    predictor::P
    consolidate::C
end

function (f::OptimalControlFunction)(p::AbstractVector{T}, ::Any) where {T}
    (; f_oop, predictor, consolidate) = f
    trajectory, ps = predict(predictor, p)
    f_oop(vec(trajectory), ps, zero(eltype(p))) |> consolidate
end

function (f::OptimalControlFunction)(u::AbstractVector, p::AbstractVector{T}, ::Any) where {T}
    (; f_iip, predictor) = f
    trajectory, ps = predict(predictor, p)
    f_iip(u, vec(trajectory), ps, zero(eltype(p)))
end


function OptimalControlFunction{IIP}(ex, prob, alg::SciMLBase.DEAlgorithm, args...; consolidate=identity, kwargs...) where {IIP}
    (; system, substitutions) = prob
    t = ModelingToolkit.get_iv(system)
    vars = operation.(ModelingToolkit.unknowns(system))
    empty!(substitutions)
    new_ex = map(ex) do eq
        collect_explicit_timepoints!(substitutions, eq, vars, t)
    end
    statevars, cost_substitutions, saveat = create_cost_substitutions(substitutions, vars)
    new_ex = map(new_ex) do eq
        substitute(eq, cost_substitutions)
    end
    tspan = extrema(saveat)
    predictor = OCPredictor{IIP}(system, alg, tspan, args...; saveat = saveat, kwargs...)
    foop, fiip = generate_custom_function(system, new_ex, vec(statevars[1]); expression=Val{false}, kwargs...)
    return OptimalControlFunction{
        typeof(foop),typeof(fiip),typeof(predictor),typeof(consolidate)
    }(foop, fiip, predictor, consolidate)
end

function create_cost_substitutions(subs, vars)
    timepoints = first.(values(subs))
    sort!(timepoints)
    unique!(timepoints)
    @info timepoints
    statevars = @variables $(gensym(:X))[1:length(vars), 1:length(timepoints)]
    newsubs = Dict()
    for (k, v) in subs
        opidx = findfirst(Base.Fix1(Base.isequal, operation(k)), vars)
        newsubs[last(v)] = getindex(statevars[1], opidx, findfirst(==(first(v)), timepoints))
    end
    return statevars, newsubs, timepoints
end

function expand_timepoints!(prob::OCProblemBuilder)
    (; system, substitutions) = prob
    t = ModelingToolkit.get_iv(system)
    vars = operation.(ModelingToolkit.unknowns(system))
    costs = ModelingToolkit.get_costs(system)
    constraints = ModelingToolkit.constraints(system)
    empty!(substitutions)
    new_costs = map(costs) do eq
        collect_explicit_timepoints!(substitutions, eq, vars, t)
    end
    cost_substitutions = create_cost_substitutions(substitutions, vars)
    empty!(substitutions)
    new_constraints = map(constraints) do con
        con = Symbolics.canonical_form(con)
        new_lhs = collect_explicit_timepoints!(substitutions, con.lhs, vars, t)
        isa(con, Equation) ? new_lhs ~ 0 : new_lhs ≲ con.rhs
    end
    constraint_substitutions = create_cost_substitutions(substitutions, vars)
    sys = @set system.costs = new_costs
    sys = @set sys.constraints = new_constraints
    return @set prob.system = sys
end


