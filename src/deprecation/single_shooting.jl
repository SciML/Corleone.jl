include("controls.jl")

# Keep initialization callbacks at the whole-vector level: calling one per scalar
# parameter would change the order and number of draws from the supplied RNG.
struct LegacyInitializer{L, F} <: LuxCore.AbstractLuxWrapperLayer{:layer}
    layer::L
    initialize::F
end

LuxCore.parameterlength(x::LegacyInitializer{<:InitialCondition}) = LuxCore.parameterlength(x.layer.tunable)

LuxCore.initialparameters(rng::Random.AbstractRNG, x::LegacyInitializer) = x.initialize(rng)
LuxCore.initialstates(rng::Random.AbstractRNG, x::LegacyInitializer) = LuxCore.initialstates(rng, x.layer)
(x::LegacyInitializer)(input, ps, st) = x.layer(input, ps, st)
get_timestops(x::LegacyInitializer, ps, st, T) = get_timestops(x.layer, ps, st, T)

legacy_default_u0(rng, prob, indices, bounds) = clamp.(prob.u0[indices], first(bounds)[indices], last(bounds)[indices])
function legacy_default_p0(rng, prob, indices, bounds)
    vals, _... = SciMLStructures.canonicalize(SciMLStructures.Tunable(), prob.p)
    return clamp.(vals[indices], bounds...)
end

function legacy_check_bounds(bounds, n, description)
    length(first(bounds)) == length(last(bounds)) == n || throw(ArgumentError("Inconsistent $description bounds"))
    all(first(bounds) .<= last(bounds)) || throw(ArgumentError("Inconsistent $description bounds"))
    return bounds
end

"""
    SingleShootingLayer(prob, alg; controls=[], tunable_ic=Int[], bounds_ic=nothing,
        state_initialization, bounds_p=nothing, parameter_initialization,
        quadrature_indices=Int[], kwargs...)

Translate the legacy single-shooting constructor into a [`ShootingLayer`](@ref).
`controls` contains `parameter_index => ControlParameter(...)` pairs. Initial
conditions outside `tunable_ic` remain fixed; non-control parameters remain tunable.
Extra keywords remake the underlying problem before translation.

The result uses **the new Lux parameter/state layout and ShootingLayer output**;
this function does not restore the old trajectory or bounds API. Bounds are used
for default initialization only. Custom initializers retain the legacy signatures
`(rng, problem, indices, bounds)` and run during `LuxCore.setup`.
`quadrature_indices` are validated; they need no special treatment in a single shoot.
"""
function SingleShootingLayer(
        prob::SciMLBase.AbstractDEProblem, alg::SciMLBase.AbstractSciMLAlgorithm;
        controls = [], tunable_ic = Int[], bounds_ic = nothing,
        state_initialization = legacy_default_u0, bounds_p = nothing,
        parameter_initialization = legacy_default_p0, quadrature_indices = Int[],
        kwargs...,
    )
    prob = remake(prob; kwargs...)
    vals, _... = SciMLStructures.canonicalize(SciMLStructures.Tunable(), prob.p)
    controls = collect(controls)
    indices = Int[first(c) for c in controls]
    tunable_ic = Int[tunable_ic...]
    allunique(indices) || throw(ArgumentError("Control parameter indices must be unique"))
    allunique(tunable_ic) || throw(ArgumentError("Tunable initial-condition indices must be unique"))
    all(i -> checkbounds(Bool, vals, i), indices) || throw(ArgumentError("Invalid control parameter index"))
    all(i -> checkbounds(Bool, prob.u0, i), tunable_ic) || throw(ArgumentError("Invalid tunable initial-condition index"))
    all(i -> checkbounds(Bool, prob.u0, i), quadrature_indices) || throw(ArgumentError("Invalid quadrature index"))
    p_indices = setdiff(eachindex(vals), indices)
    ic_bounds = isnothing(bounds_ic) ? (fill(-Inf, size(prob.u0)), fill(Inf, size(prob.u0))) : bounds_ic
    p_bounds = isnothing(bounds_p) ? (fill(-Inf, length(p_indices)), fill(Inf, length(p_indices))) : bounds_p
    legacy_check_bounds(ic_bounds, length(prob.u0), "initial-condition")
    legacy_check_bounds(p_bounds, length(p_indices), "parameter")

    grids = map(controls) do (_, control)
        issorted(control.t) && allunique(control.t) || throw(ArgumentError("Control grids must be sorted and unique"))
        grid = collect(get_timegrid(control, prob.tspan))
        isempty(grid) && throw(ArgumentError("Each control needs a time point inside the problem time span"))
        grid
    end
    # The legacy first value is active at tspan[1], even when its grid starts later.
    # New PiecewiseConstant stores only the boundaries between successive values.
    definitions = [indices[i] => grids[i][2:end] for i in eachindex(indices)]
    fixed_ic = setdiff(eachindex(prob.u0), tunable_ic)
    layer = ShootingLayer(prob, alg, first(prob.tspan) => (; fixed = fixed_ic), definitions...)
    ic = only(layer.initialcondition.initialconditions)
    initialize_ic = function (rng)
        values = state_initialization(rng, prob, tunable_ic, ic_bounds)
        length(values) == length(tunable_ic) || throw(DimensionMismatch("Initial-condition initializer returned the wrong number of values"))
        # InitialCondition stores tunables in state-index order, rather than the
        # order supplied by the legacy caller.
        return NamedTuple{keys(ic.tunable)}(Tuple(values[sortperm(tunable_ic)]))
    end
    conditions = InitialConditions(
        layer.initialcondition.name, layer.initialcondition.lookup,
        (LegacyInitializer(ic, initialize_ic),),
    )
    parameter = layer.parameter
    initialize_parameter = function (rng)
        p = parameter_initialization(rng, prob, p_indices, p_bounds)
        length(p) == length(p_indices) || throw(DimensionMismatch("Parameter initializer returned the wrong number of values"))
        control_values = map(controls) do (_, control)
            values = get_controls(rng, control; tspan = prob.tspan)
            length(values) == control_length(control; tspan = prob.tspan) || throw(DimensionMismatch("Control initializer returned the wrong number of values"))
            values
        end
        parameters = map(enumerate(values(parameter.parameters))) do (i, pc)
            c = findfirst(==(i), indices)
            v = isnothing(c) ? [p[findfirst(==(i), p_indices)]] : control_values[c]
            (; lookup = (;), parameter = (; ps = v))
        end
        return (; parameters = NamedTuple{keys(parameter.parameters)}(Tuple(parameters)))
    end
    return ShootingLayer(layer.name, prob, alg, conditions, LegacyInitializer(parameter, initialize_parameter))
end
