# Measurement-based design adapted from Christoph Plate (@chplate)'s fork.
# See README.md for the original commits and license attribution.

"""
$(TYPEDEF)

A shooting layer for local optimal experimental design. The augmented state holds
physical states, parameter sensitivities, and the upper triangle of the continuous
Fisher information matrix. Discrete observations are accumulated on demand.

Call `OEDLayer(problem, algorithm, definitions...; params, measurements, kwargs...)`
or wrap an existing `ShootingLayer`. Unspecified physical parameters are fixed at
their nominal values; explicit annotations select design variables. All measurement
weights are tunable in `[0, 1]`. `params` selects the physical parameters whose
sensitivities define the information matrix, independently of their tunability.

Lux parameters retain the shooting layer's layout. State additionally stores
`F_init`, a prior information matrix, included once by `fisher_information`.
Evaluation returns `(trajectory, next_st)` and preserves derivatives through
current parameters and numerical state. Full physical-state shooting resets also
lift the sensitivity and information states, with ordinary continuity constraints.

# Fields
$(FIELDS)
"""
struct OEDLayer{L, P, M, I, B} <: LuxCore.AbstractLuxWrapperLayer{:shooting}
    "The augmented shooting layer."
    shooting::L
    "The augmented differential equation problem, shared with `shooting.problem`."
    augmented_prob::P
    "Discrete/continuous measurement specifications and numerical observation getters."
    measurements::M
    "Original physical parameter indices in sensitivity-column order."
    params::I
    "The fixed parameter-direction basis; identity without SVD reduction."
    basis::B
end

function annotation_indices(sys, definition, kind)
    selector = first(definition)
    if selector isa Corleone.AbstractAnnotation
        is_variable(selector) == (kind === :variable) || return Int[]
        selector = isnothing(selector.index) ? selector.sym : selector.index
    elseif kind === :variable
        return Int[]
    end
    selectors = selector isa Union{Tuple, AbstractVector} ? selector : (selector,)
    indexof = kind === :variable ? variable_index : parameter_index
    return map(collect(selectors)) do key
        index = key isa Integer ? key : indexof(sys, key)
        index isa Integer || throw(ArgumentError("Unknown $kind selector: $key"))
        Int(index)
    end
end

function OEDLayer(
        problem::SciMLBase.AbstractDEProblem, algorithm::SciMLBase.AbstractDEAlgorithm,
        definitions...; params = eachindex(problem.p), measurements = (),
        fixed_parameters = nothing, controls = (), quadratures = (), bounds = nothing,
        ensemblealg = EnsembleSerial(), svd = false, ns = nothing,
        threshold_singular_values = 0.95, threshold_singular_vectors = 0.1,
    )
    sys = ControlSymbolCache(problem)
    selected = [key isa Integer ? Int(key) : parameter_index(sys, key) for key in params]
    !isempty(selected) && allunique(selected) && all(i -> i isa Integer && i in eachindex(problem.p), selected) ||
        throw(ArgumentError("Select valid, unique physical parameter indices or symbols."))
    selected = Int.(selected)
    normalized = map(m -> measurement_schedule(m, problem.tspan), Tuple(measurements))
    ids = map(m -> m.id, normalized)
    allunique(ids) && all(id -> !is_parameter(sys, id) && !is_variable(sys, id), ids) ||
        throw(ArgumentError("Measurement identifiers must be unique and distinct from model symbols."))
    discrete = filter(m -> m isa DiscreteMeasurement, normalized)
    continuous = filter(m -> m isa ContinuousMeasurement, normalized)
    augmented, observed, basis = augment_system(
        problem, algorithm; params = selected, continuous_measurements = continuous,
        discrete_measurements = discrete, svd, ns, threshold_singular_values,
        threshold_singular_vectors,
    )
    points = sort!(unique!(reduce(vcat, map(m -> m.tpoints, discrete); init = eltype(problem.tspan)[])))
    if !isempty(points)
        original = get(problem.kwargs, :saveat, eltype(points)[])
        saveat = original isa Number ? collect(first(problem.tspan):original:last(problem.tspan)) : collect(original)
        augmented = remake(augmented; saveat = sort!(unique!(vcat(saveat, points))))
    end
    weights = map((discrete..., continuous...)) do measurement
        Tunable(measurement.id; default = 1.0, lowerbound = 0.0, upperbound = 1.0) =>
            measurement_switches(measurement, problem.tspan)
    end
    defined = reduce(vcat, map(def -> annotation_indices(sys, def, :parameter), definitions); init = Int[])
    fixed = isnothing(fixed_parameters) ? setdiff(eachindex(problem.p), defined) : fixed_parameters
    augmented_definitions = Pair[definitions...]
    # Inspect schedules without running initializers or drawing from an RNG.
    resets = filter(def -> !isempty(annotation_indices(sys, def, :variable)), definitions)
    for definition in resets
        if !Corleone.is_tunable(first(definition)) && !isnothing(last(definition)) &&
                any(t -> first(problem.tspan) < t <= last(problem.tspan), last(definition))
            throw(ArgumentError("Prescribed physical resets after the initial time need sensitivity reset maps; use tunable matching resets for shooting."))
        end
    end
    times = sort!(unique!(reduce(vcat, [collect(last(def)) for def in resets if !isnothing(last(def))]; init = eltype(problem.tspan)[])))
    full_resets = filter(times) do time
        first(problem.tspan) < time < last(problem.tspan) && all(eachindex(problem.u0)) do index
            any(def -> !isnothing(last(def)) && time in last(def) && index in annotation_indices(sys, def, :variable), resets)
        end
    end
    if !isempty(full_resets)
        extra = (length(problem.u0) + 1):length(augmented.u0)
        push!(augmented_definitions, Tunable(collect(extra); kind = Val(:variable), default = zeros(length(extra))) => full_resets)
    end
    cache = ControlSymbolCache(augmented; controls = (controls..., ids...), quadratures, bounds = Dict(Symbol(key) => interval for (key, interval) in (isnothing(bounds) ? default_bounds(sys) : bounds)))
    # Pass the cache through the problem so original annotation bounds remain available.
    f = remake(augmented.f; sys = cache)
    augmented = remake(augmented; f)
    shooting = ShootingLayer(augmented, algorithm, augmented_definitions..., weights...; fixed_parameters = fixed, ensemblealg)
    return OEDLayer(shooting, augmented, (; observed, discrete, continuous), selected, basis)
end

function shooting_definitions(layer::ShootingLayer)
    definitions = Pair[]
    for (kind, collection, indices) in (
            (Val(:parameter), layer.parameter.parameters, layer.parameter.indices),
            (Val(:variable), layer.initialcondition.initialconditions, layer.initialcondition.indices),
        )
        for (group, entry) in enumerate(values(collection))
            fixed = entry isa Corleone.FixedLayer
            child = fixed ? entry.layer : entry
            child isa Corleone.PassThrough && continue
            grid = child.lookup.collection
            grid = grid isa AbstractDict ? sort!(collect(keys(grid)); by = time -> grid[time]) : grid
            members = child.parameter isa NamedTuple ? values(child.parameter) : (child.parameter,)
            for (index, member) in zip(indices[group], members)
                held = fixed || member isa Corleone.FixedLayer
                parameter = member isa Corleone.FixedLayer ? member.layer : member
                symbols = kind === Val(:variable) ? variable_symbols(layer.sys) : parameter_symbols(layer.sys)
                indexof = kind === Val(:variable) ? variable_index : parameter_index
                sym = only(filter(sym -> indexof(layer.sys, sym) == index, symbols))
                lo, hi = get_bounds(layer.sys, sym)
                constructor = held ? Constant : Tunable
                push!(definitions, constructor(index; kind, default = parameter.init_parameters, lowerbound = lo, upperbound = hi) => grid)
            end
        end
    end
    return definitions
end

function OEDLayer(layer::ShootingLayer; kwargs...)
    return OEDLayer(
        layer.problem, layer.alg, shooting_definitions(layer)...;
        controls = control_symbols(layer.sys), quadratures = quadrature_symbols(layer.sys),
        bounds = default_bounds(layer.sys), ensemblealg = layer.ensemblealg, kwargs...
    )
end

LuxCore.initialparameters(rng::Random.AbstractRNG, oed::OEDLayer) = LuxCore.initialparameters(rng, oed.shooting)
function LuxCore.initialstates(rng::Random.AbstractRNG, oed::OEDLayer)
    n = size(oed.basis, 2)
    return merge(LuxCore.initialstates(rng, oed.shooting), (; F_init = zeros(eltype(oed.augmented_prob.u0), n, n)))
end
LuxCore.statelength(oed::OEDLayer) = LuxCore.statelength(oed.shooting) + size(oed.basis, 2)^2
(oed::OEDLayer)(x, ps, st) = oed.shooting(x, ps, st)
SymbolicIndexingInterface.symbolic_container(oed::OEDLayer) = symbolic_container(oed.shooting)
get_size_F(oed::OEDLayer) = (size(oed.basis, 2), size(oed.basis, 2))
n_observed(oed::OEDLayer) = length(oed.measurements.continuous) + length(oed.measurements.discrete)
Corleone.get_number_of_shooting_constraints(oed::OEDLayer, ps, st) = get_number_of_shooting_constraints(oed.shooting, ps, st)
Corleone.add_parameter_segments!(oed::OEDLayer, ps, st) = add_parameter_segments!(oed.shooting, ps, st)

sample_index(traj, t) = begin
    i = searchsortedlast(traj.t, t)
    i > 0 && traj.t[i] == t || throw(ArgumentError("Measurement time $t was not saved."))
    i
end
@non_differentiable sample_index(traj, t)
weight_at(oed, traj, id, time) = SymbolicIndexingInterface.parameter_values_at_time(traj, traj, time)[parameter_index(traj, id)]

function discrete_information(oed::OEDLayer, traj::Trajectory)
    measurements = oed.measurements.discrete
    isempty(measurements) && return zeros(eltype(first(traj.u)), get_size_F(oed))
    values = oed.measurements.observed.hx_G_discrete(traj)
    ranges = oed.measurements.observed.discrete_ranges
    contributions = map(measurements, ranges) do measurement, block
        sum(measurement.tpoints) do time
            J = values[sample_index(traj, time)][block, :]
            weight_at(oed, traj, measurement.id, time) * (J' * J)
        end
    end
    return sum(contributions)
end

"""
$(SIGNATURES)

Return continuous plus discrete Fisher information from an existing trajectory,
without solving again. Weights multiply information linearly. Include `st.F_init`
once unless `add_initial=false`. Layer calls return `(F, next_st)`.
"""
function fisher_information(oed::OEDLayer, traj::Trajectory; add_initial = true)
    F = last(oed.measurements.observed.fisher(traj)) + discrete_information(oed, traj)
    return add_initial ? F + traj.st.F_init : F
end
function fisher_information(oed::OEDLayer, x, ps, st; add_initial = true)
    traj, next_st = oed(x, ps, st)
    return fisher_information(oed, traj; add_initial), next_st
end

"""
$(SIGNATURES)

Return the saved physical-state sensitivity matrices, with one column per selected
parameter direction. With `svd=true`, columns use `oed.basis`. The four-argument
form solves once and returns `(matrices, next_st)`.
"""
sensitivities(oed::OEDLayer, traj::Trajectory) = oed.measurements.observed.sensitivities(traj)
function sensitivities(oed::OEDLayer, x, ps, st)
    traj, next_st = oed(x, ps, st)
    return sensitivities(oed, traj), next_st
end
"""
$(SIGNATURES)

Return measurement outputs at each saved state time, ordered continuous then
discrete. Vector-valued measurements contribute consecutive entries. This does
not select discrete measurement events; use each measurement's `tpoints` for those.
The four-argument form returns `(outputs, next_st)`.
"""
observed_equations(oed::OEDLayer, traj::Trajectory) = map(
    vcat,
    oed.measurements.observed.observed_continuous(traj),
    oed.measurements.observed.observed_discrete(traj),
)
function observed_equations(oed::OEDLayer, x, ps, st)
    traj, next_st = oed(x, ps, st)
    return observed_equations(oed, traj), next_st
end

"""
$(SIGNATURES)

Return the sum of the event weights for each discrete measurement. Pass a
trajectory, or a callable `lookup(id, time)`. Measurements retain their declared order.
"""
function discrete_sampling_sums(oed::OEDLayer, lookup)
    return collect(
        map(oed.measurements.discrete) do m
            sum(time -> lookup(m.id, time), m.tpoints)
        end
    )
end
"""
$(SIGNATURES)

Integrate each continuous measurement's piecewise-constant weight over the full
problem time span. Pass a trajectory, or `lookup(id, time)` and the sorted interval
boundaries. Measurements retain their declared order.
"""
function continuous_sampling_sums(oed::OEDLayer, lookup, times)
    return collect(
        map(oed.measurements.continuous) do m
            sum(i -> (times[i + 1] - times[i]) * lookup(m.id, times[i]), 1:(length(times) - 1))
        end
    )
end
function sampling_times(oed, ps, st)
    return sort!(
        unique!(
            vcat(
                collect(oed.augmented_prob.tspan),
                Corleone.get_timestops(oed.shooting.parameter, nothing, st.parameter, eltype(oed.augmented_prob.tspan))
            )
        )
    )
end
@non_differentiable sampling_times(oed, ps, st)
discrete_sampling_sums(oed::OEDLayer, traj::Trajectory) = discrete_sampling_sums(oed, (id, time) -> weight_at(oed, traj, id, time))
continuous_sampling_sums(oed::OEDLayer, traj::Trajectory) = continuous_sampling_sums(oed, (id, time) -> weight_at(oed, traj, id, time), sampling_times(oed, traj.ps, traj.st))

"""
$(SIGNATURES)

Return sampling budgets, continuous measurements first and discrete measurements
second. Continuous budgets are time integrals; discrete budgets are sums at
measurement events. The four-argument form reads `ps` and `st` without solving
the differential equation. Multi-experiment budgets concatenate experiment order.
"""
function sampling_sums(oed::OEDLayer, traj::Trajectory)
    lookup = (id, time) -> weight_at(oed, traj, id, time)
    times = sampling_times(oed, traj.ps, traj.st)
    return vcat(continuous_sampling_sums(oed, lookup, times), discrete_sampling_sums(oed, lookup))
end
function sampling_sums(oed::OEDLayer, x, ps, st)
    lookup = (id, time) -> first(oed.shooting.parameter(time, ps.parameter, st.parameter))[parameter_index(oed.shooting, id)]
    times = sampling_times(oed, ps, st)
    return vcat(continuous_sampling_sums(oed, lookup, times), discrete_sampling_sums(oed, lookup))
end

"""
$(SIGNATURES)

Write `sampling_sums(layer, args...)` into the prefix of `res` and return `res`.
The buffer must have at least one entry per measurement; its remaining entries
are unchanged.
"""
function sampling_sums!(res, oed::OEDLayer, args...)
    length(res) >= n_observed(oed) || throw(DimensionMismatch("Expected $(n_observed(oed)) sampling budgets."))
    copyto!(res, sampling_sums(oed, args...))
    return res
end
get_sampling_sums(args...) = sampling_sums(args...)
get_sampling_sums!(args...) = sampling_sums!(args...)

"""
$(SIGNATURES)

Return state with the information from completed experiments added to `F_init`.
Each experiment supplies `ps` and may supply its own `st` (otherwise use `st`).
Existing priors are excluded from each experiment's contribution to avoid counting
them repeatedly. The two-argument form starts from freshly initialized state.
An empty collection leaves state unchanged.
"""
function update_fim(oed::OEDLayer, experiments, st)
    isempty(experiments) && return st
    added = sum(experiments) do experiment
        local_st = hasproperty(experiment, :st) ? experiment.st : st
        first(fisher_information(oed, nothing, experiment.ps, local_st; add_initial = false))
    end
    return merge(st, (; F_init = st.F_init + added))
end
update_fim(oed::OEDLayer, experiments) = update_fim(oed, experiments, LuxCore.initialstates(Random.default_rng(), oed))

"""
$(SIGNATURES)

Solve once and return `(gains, next_st)`. Each saved time contains the unweighted
outer products `J[i:i, :]' * J[i:i, :]` of individual observation-sensitivity rows,
continuous observations first, then discrete observations. These are per output
row, even when several rows share one measurement weight.
"""
function local_information_gain(oed::OEDLayer, x, ps, st)
    traj, next_st = oed(x, ps, st)
    values = oed.measurements.observed.local_information_gain(traj)
    return map(J -> [J[i:i, :]' * J[i:i, :] for i in axes(J, 1)], values), next_st
end
"""
$(SIGNATURES)

Solve once and return `(gains, next_st)`. Each saved time contains per output-row
matrices `(J[i:i, :] * inv(F))' * (J[i:i, :] * inv(F))`. Here `F` includes the prior
and all continuous/discrete information and must be nonsingular. Their traces
measure the marginal improvement in A-optimality before interval/event weighting.
"""
function global_information_gain(oed::OEDLayer, x, ps, st)
    traj, next_st = oed(x, ps, st)
    inverse = inv(fisher_information(oed, traj))
    values = oed.measurements.observed.local_information_gain(traj)
    return map(J -> [(J[i:i, :] * inverse)' * (J[i:i, :] * inverse) for i in axes(J, 1)], values), next_st
end

# Keep the mask in the native Lux tree and let the chosen vectorizer flatten it.
function sampling_integrality!(mask, layer::OEDLayer)
    ids = (map(m -> m.id, layer.measurements.discrete)..., map(m -> m.id, layer.measurements.continuous)...)
    for (name, pc) in pairs(layer.shooting.parameter.parameters)
        pc isa Corleone.FixedLayer && continue
        child = getproperty(mask.parameter.parameters, name).parameter
        if pc.parameter isa NamedTuple
            for id in keys(pc.parameter)
                id in ids && (getproperty(child, id).ps .= true)
            end
        elseif name in ids
            child.ps .= true
        end
    end
    return mask
end
