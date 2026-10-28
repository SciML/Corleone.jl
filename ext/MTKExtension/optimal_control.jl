"""
$(SIGNATURES)

Build a dynamic optimization problem from an uncompiled ModelingToolkit `System`.
System `costs` are summed. Definite time integrals introduce quadrature states;
timed state/control reads use the shared trajectory expression evaluator.
Equations become zero equalities and inequalities become residuals bounded above
by zero. State and parameter bounds come from MTK metadata.
States marked `tunable = true` receive a free initial value at `tspan[1]`.

Pass controls as `input => timegrid`. A leading `tspan[1]` is accepted for
compatibility with interval-start grids; remaining entries are interior switches.
`shooting` selects full-state resets strictly inside the time span. With no
explicit `tspan`, infer it from integral endpoints, timed reads, and supplied grids.
The system is compiled with `split = false` to use flat numerical parameters.
No MTK initialization system is built; initial values must be supplied or have
numerical defaults.

`problem_kwargs` are forwarded to `ODEProblem`; remaining keywords configure
the numerical `CorleoneDynamicOptProblem` facade (for example `evaluator`).
"""
function Corleone.CorleoneDynamicOptProblem(
        sys::ModelingToolkit.System, inits::AbstractVector, controls::Pair...;
        algorithm::SciMLBase.AbstractDEAlgorithm,
        shooting::Union{AbstractVector{<:Real}, Nothing} = nothing,
        tspan = nothing, ensemblealg = SciMLBase.EnsembleSerial(),
        sensealg = SciMLBase.NoAD(), problem_kwargs = (;), rng = default_rng(),
        kwargs...
    )
    iv = ModelingToolkit.get_iv(sys)
    accumulations = Dict()
    costs = map(c -> lower_integrals!(accumulations, c, iv), ModelingToolkit.get_costs(sys))
    isempty(costs) && throw(ArgumentError("The system must declare at least one cost."))
    constraints = map(ModelingToolkit.get_constraints(sys)) do constraint
        canonical = Symbolics.canonical_form(constraint)
        lower_integrals!(accumulations, canonical.lhs, iv)
    end
    upper = zeros(length(constraints))
    lower = map(c -> c isa Symbolics.Inequality ? -Inf : 0.0, ModelingToolkit.get_constraints(sys))
    if !isempty(accumulations)
        sys = ModelingToolkit.add_accumulations(
            sys, [q => only(SymbolicUtils.arguments(integral)) for (integral, q) in accumulations]
        )
    end

    inputs = collect(first.(controls))
    compiled = ModelingToolkit.mtkcompile(sys; inputs, split = false)
    any(sym -> symbolic_type(sym) isa ArraySymbolic, parameter_symbols(compiled)) &&
        throw(ArgumentError("The MTK frontend currently requires scalar parameters. Declare separate scalar parameters instead of parameter arrays; MTK's split=false problems do not support array parameters."))
    signals = Dict(
        Symbolics.unwrap(sym) => (Symbol(sym), kind)
            for (kind, symbols) in ((:variable, variable_symbols(compiled)), (:parameter, parameter_symbols(compiled)))
            for sym in symbols
    )
    times = Real[]
    # Bare parameters are terminal reads. Fill in their time after inference.
    terminal = isnothing(tspan) ? 0.0 : last(tspan)
    objective = symbolic_expression(sum(costs), signals, terminal, times)
    expressions = map(c -> symbolic_expression(c, signals, terminal, times), constraints)
    foreach(control -> append!(times, last(control)), controls)
    isnothing(shooting) || append!(times, shooting)
    if isnothing(tspan)
        length(unique(times)) >= 2 || throw(ArgumentError("Supply tspan or at least two distinct expression/grid times."))
        tspan = extrema(times)
        objective = symbolic_expression(sum(costs), signals, last(tspan), Real[])
        expressions = map(c -> symbolic_expression(c, signals, last(tspan), Real[]), constraints)
    end
    lo, hi = tspan
    lo < hi || throw(ArgumentError("tspan must be increasing."))
    all(t -> lo <= t <= hi, times) || throw(ArgumentError("Expression and schedule times must lie within tspan."))
    prob = SciMLBase.ODEProblem(
        compiled, inits, tspan; check_compatibility = false, sensealg,
        problem_kwargs..., build_initializeprob = false
    )
    resets = isnothing(shooting) ? typeof(lo)[] : sort!(unique!(filter(t -> lo < t < hi, collect(shooting))))
    annotations = Corleone.Annotations(
        compiled; u0 = prob.u0, p = prob.p, controls = inputs,
        lift_states = !isempty(resets)
    )
    bounds = Dict(
        annotation.sym => (annotation.lowerbound, annotation.upperbound)
            for collection in values(annotations) for annotation in collection
    )
    definitions = Pair[]
    for annotation in annotations.parameters
        position = findfirst(control -> isequal(Symbolics.unwrap(first(control)), annotation.sym), controls)
        grid = if isnothing(position)
            nothing
        else
            supplied = last(controls[position])
            issorted(supplied) && allunique(supplied) || throw(ArgumentError("Control grids must be sorted and unique."))
            switches = collect(filter(!=(lo), supplied))
            all(t -> lo < t < hi, switches) ||
                throw(ArgumentError("Control grids must have interior switches, optionally preceded by tspan[1]."))
            switches
        end
        push!(definitions, annotation => grid)
    end
    for annotation in annotations.variables
        grid = ModelingToolkit.istunable(annotation.sym, false) ? vcat([lo], resets) : resets
        isempty(grid) && continue
        push!(definitions, annotation => grid)
    end
    layer = ShootingLayer(
        prob, algorithm, definitions...; ensemblealg,
        controls = inputs, quadratures = collect(values(accumulations)), bounds
    )
    return Corleone.CorleoneDynamicOptProblem(
        layer, expression_wrapper(objective), map(expression_wrapper, expressions)...;
        lcons = lower, ucons = upper, rng, kwargs...
    )
end
