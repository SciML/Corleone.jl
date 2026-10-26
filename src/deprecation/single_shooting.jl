"""
$(SIGNATURES)

Deprecated single-shooting constructor returning a [`ShootingLayer`](@ref).
Translate `controls = [index => ControlParameter(...), ...]`, `tunable_ic`,
`bounds_ic`, `bounds_p`, and `quadrature_indices` into native annotations and
symbolic bookkeeping. Initial conditions in `tunable_ic` are tunable at the problem
start; other initial conditions pass through `problem.u0`. Non-control parameters
remain tunable. Extra keywords remake the differential equation problem.

The return value uses the new Lux parameter/state layout, with residuals in
`trajectory.residuals`. Legacy whole-vector initialization callbacks are not
supported; set annotation defaults using count or timepoint callbacks instead.
"""
function SingleShootingLayer(prob::SciMLBase.AbstractDEProblem,
        alg::SciMLBase.AbstractSciMLAlgorithm; kwargs...)
    Base.depwarn("SingleShootingLayer is deprecated; use ShootingLayer with Tunable/Constant annotations. The returned layer uses the new parameter/state layout and trajectory.residuals.", :SingleShootingLayer)
    return legacy_shooting_layer(prob, alg; kwargs...)
end

function legacy_shooting_layer(prob, alg; controls = (), tunable_ic = Int[],
        bounds_ic = nothing, bounds_p = nothing, quadrature_indices = Int[],
        state_initialization = nothing, parameter_initialization = nothing,
        name = nothing, kwargs...)
    isnothing(state_initialization) && isnothing(parameter_initialization) ||
        throw(ArgumentError("Legacy whole-vector initialization callbacks are unsupported. Use annotation default vectors or (rng, n::Int)/(rng, times::AbstractVector) initializers."))
    prob = remake(prob; kwargs...)
    sys = get_symbolic_container(prob)
    pvals, _, _ = SciMLStructures.canonicalize(SciMLStructures.Tunable(), prob.p)
    uvals, _, _ = SciMLStructures.canonicalize(SciMLStructures.Tunable(), prob.u0)
    conditions = collect(tunable_ic)
    allunique(conditions) && all(i -> i isa Integer && checkbounds(Bool, uvals, i), conditions) ||
        throw(ArgumentError("Tunable initial-condition indices must be valid and unique."))
    qindices = collect(quadrature_indices)
    allunique(qindices) && all(i -> i isa Integer && checkbounds(Bool, uvals, i), qindices) ||
        throw(ArgumentError("Quadrature indices must be valid and unique."))
    definitions = Pair[]
    control_indices = Int[]
    for entry in controls
        entry isa Pair && last(entry) isa Pair && first(last(entry)) isa AbstractAnnotation ||
            throw(ArgumentError("Legacy controls must be parameter_index => (annotation => switching_times) pairs, as returned by ControlParameter."))
        selector, definition = entry
        index = selector isa Integer ? selector : parameter_index(sys, selector)
        index isa Integer && checkbounds(Bool, pvals, index) ||
            throw(ArgumentError("Invalid control parameter: $selector"))
        push!(control_indices, index)
        annotation, grid = definition
        is_variable(annotation) && throw(ArgumentError("A control must select a parameter."))
        # The first selected interval is active at the problem start. Endpoints
        # do not introduce zero-length segments or unused final control values.
        times = filter(t -> first(prob.tspan) < t < last(prob.tspan), grid)
        annotation = remake(annotation; index, sym = nothing, kind = Val(:parameter))
        push!(definitions, annotation => times)
    end
    allunique(control_indices) || throw(ArgumentError("Define each control parameter only once."))
    ordinary = setdiff(eachindex(pvals), control_indices)
    plower, pupper = legacy_bounds(isnothing(bounds_p) ? (-Inf, Inf) : bounds_p,
        length(ordinary), "parameter")
    ulower, uupper = legacy_bounds(isnothing(bounds_ic) ? (-Inf, Inf) : bounds_ic,
        length(uvals), "initial-condition")
    for (k, index) in enumerate(ordinary)
        push!(definitions, Tunable(index; kind = Val(:parameter),
            default = clamp(pvals[index], plower[k], pupper[k]),
            lowerbound = plower[k], upperbound = pupper[k]) => nothing)
    end
    for index in conditions
        push!(definitions, Tunable(index; kind = Val(:variable),
            default = clamp(uvals[index], ulower[index], uupper[index]),
            lowerbound = ulower[index], upperbound = uupper[index]) => [first(prob.tspan)])
    end
    # Retain state bounds even for initial conditions that pass through u0.
    bounds = Dict(sym => (ulower[variable_index(sys, sym)], uupper[variable_index(sys, sym)])
        for sym in variable_symbols(sys))
    return ShootingLayer(prob, alg, definitions...; name, controls = control_indices,
        quadratures = qindices, bounds)
end
