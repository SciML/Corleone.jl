"""
$(SIGNATURES)

Deprecated multiple-shooting constructor returning a [`ShootingLayer`](@ref).
Accept a problem and algorithm, or an existing `ShootingLayer`, followed by shooting
points as a vector or positional arguments. Every state, including quadratures,
is lifted at interior shooting points. `ensemble_alg` selects block execution.

The returned layer uses the native per-variable parameter layout, free tunable
initial values, and interior matching residuals. Parameters and controls remain
shared across shooting blocks.
Legacy stage-parameter/control lifting and whole-layer `initialization` callbacks
are not restored; express initial guesses through annotation defaults instead.
"""
function MultipleShootingLayer(
        prob::SciMLBase.AbstractDEProblem,
        alg::SciMLBase.AbstractSciMLAlgorithm, times...;
        ensemble_alg = EnsembleSerial(), initialization = nothing, kwargs...
    )
    Base.depwarn("MultipleShootingLayer is deprecated; use ShootingLayer with scheduled variable annotations and ensemblealg. All states, including quadratures, use native continuity residuals.", :MultipleShootingLayer)
    isnothing(initialization) || throw(ArgumentError("Use annotation defaults instead of a legacy multiple-shooting initialization callback."))
    layer = legacy_shooting_layer(prob, alg; kwargs...)
    return legacy_lift_states(layer, times, ensemble_alg)
end

function MultipleShootingLayer(
        layer::ShootingLayer, times...;
        ensemble_alg = layer.ensemblealg, initialization = nothing
    )
    Base.depwarn("MultipleShootingLayer is deprecated; define scheduled variable annotations on ShootingLayer and select ensemblealg.", :MultipleShootingLayer)
    isnothing(initialization) || throw(ArgumentError("Use annotation defaults instead of a legacy multiple-shooting initialization callback."))
    return legacy_lift_states(layer, times, ensemble_alg)
end

function legacy_lift_states(layer, times, ensemblealg)
    points = length(times) == 1 && only(times) isa AbstractVector ? collect(only(times)) : collect(times)
    all(t -> t isa Real && first(layer.problem.tspan) <= t <= last(layer.problem.tspan), points) ||
        throw(ArgumentError("Shooting points must lie inside the problem time span."))
    sort!(unique!(points))
    filter!(t -> first(layer.problem.tspan) < t < last(layer.problem.tspan), points)
    values, _, _ = SciMLStructures.canonicalize(SciMLStructures.Tunable(), layer.problem.u0)
    children = map(enumerate(Base.values(layer.initialcondition.initialconditions))) do (index, ic)
        isempty(points) && return ic
        ic isa FixedLayer && throw(ArgumentError("A fixed reset schedule cannot be combined with tunable shooting resets for the same variable. Construct the desired native annotations explicitly."))
        sym = only(filter(sym -> variable_index(layer.sys, sym) == index, variable_symbols(layer.sys)))
        bounds = get_bounds(layer.sys, sym)
        fallback = clamp(values[index], bounds...)
        if ic isa PassThrough
            return InitialCondition(points, fallback; name = ic.name, bounds)
        end
        original = ic.lookup.collection
        grid = sort!(unique!(vcat(collect(keys(original)), points)))
        initialize = function (rng, times::AbstractVector)
            existing = ic.parameter.init_parameters(rng, ic.parameter.out_dims)
            return [haskey(original, t) ? existing[original[t]] : fallback for t in times]
        end
        return InitialCondition(grid, initialize; name = ic.name, bounds)
    end
    initialcondition = InitialConditions(
        layer.initialcondition.name, layer.initialcondition.repack,
        NamedTuple{keys(layer.initialcondition.initialconditions)}(Tuple(children))
    )
    return ShootingLayer(
        layer.name, layer.problem, layer.alg, ensemblealg,
        initialcondition, layer.parameter, layer.sys
    )
end
