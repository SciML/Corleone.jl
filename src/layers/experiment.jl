# Internal local-value provider. It owns no model or solver.
@concrete terse struct ExperimentLayer <: LuxCore.AbstractLuxContainerLayer{(:initialcondition, :parameter)}
    initialcondition
    parameter
    observations
end
(layer::ExperimentLayer)(::Any, ps, st) = ps, st

# Split grouped layers by semantic index before grouping each ownership domain.
# This is construction only; forward calls merge disjoint NamedTuple branches.
function experiment_entries(layer, sys, variable)
    children = variable ? layer.initialconditions : layer.parameters
    symbols = variable ? variable_symbols(sys) : parameter_symbols(sys)
    indexof = variable ? variable_index : parameter_index
    entries = []
    for (child, indices) in zip(values(children), layer.indices)
        fixed = child isa FixedLayer
        child = fixed ? child.layer : child
        if child isa PassThrough
            index = only(indices)
            sym = only(filter(sym -> indexof(sys, sym) == index, symbols))
            push!(entries, (; index, sym, name = child.name, grid = nothing, parameter = nothing, fixed = false))
            continue
        end
        collections = child.parameter isa NamedTuple ? values(child.parameter) : (child.parameter,)
        for (index, collection) in zip(indices, collections)
            leaf_fixed = fixed || collection isa FixedLayer
            collection = collection isa FixedLayer ? collection.layer : collection
            sym = only(filter(sym -> indexof(sys, sym) == index, symbols))
            transform = collection.transform isa BoundedTransform ? BoundedTransform(get_bounds(sys, sym)...) : collection.transform
            collection = ParameterCollection(collection.init_parameters, collection.out_dims, transform, collection.prior)
            push!(entries, (; index, sym, name = Symbol(sym), grid = child.lookup.collection, parameter = collection, fixed = leaf_fixed))
        end
    end
    return sort!(entries; by = entry -> entry.index)
end

function experiment_collection(template, entries, constructor; separate)
    groups = group_scalar_collections(entries; separate)
    children = NamedTuple(
        Tuple(
            map(groups) do group
                entry = first(group)
                isnothing(entry.parameter) && return entry.name => PassThrough(entry.name)
                shared_grid_layer(constructor, group, LookUp(deepcopy(entry.grid), nothing))
            end
        )
    )
    indices = Tuple([entry.index for entry in group] for group in groups)
    return constructor === InitialCondition ? InitialConditions(template.name, template.repack, children, indices) :
        Parameter(template.name, template.repack, children, indices)
end


function local_experiment_entries(layer, definitions, variables, parameters)
    definitions = definitions isa Pair ? (definitions,) : definitions
    resolved = reduce(
        vcat, map(definitions) do definition
            definition isa Pair && first(definition) isa AbstractAnnotation ||
                throw(ArgumentError("Experiment definitions must be annotation => timegrid pairs."))
            kind = is_variable(first(definition)) ? Val(:variable) : Val(:parameter)
            process_definition(layer.sys, kind, definition...)
        end; init = Pair[]
    )
    identities = [(is_variable(first(definition)), get_index(first(definition))) for definition in resolved]
    allunique(identities) || throw(ArgumentError("Define each experiment-local quantity only once."))
    locals = []
    for (annotation, grid) in resolved
        variable, index = is_variable(annotation), get_index(annotation)
        if !isnothing(grid)
            allunique(grid) || throw(ArgumentError("Experiment times must be unique."))
            all(t -> first(layer.problem.tspan) <= t <= last(layer.problem.tspan), grid) ||
                throw(ArgumentError("Experiment times must lie within the problem time span."))
            variable || issorted(grid) || throw(ArgumentError("Parameter switching times must be sorted."))
        end
        entries = variable ? variables : parameters
        entry = only(filter(entry -> entry.index == index, entries))
        grid = variable && isnothing(grid) ? Float64[] : grid
        times = variable ? grid : isnothing(grid) ? [first(layer.problem.tspan)] : vcat(first(layer.problem.tspan), grid)
        default, bounds = get_default(annotation), get_bounds(annotation)
        initialize = let default = default, times = times, bounds = bounds
            (rng, n) -> initialize_annotation_values(default, rng, n, times, bounds)
        end
        parameter = ParameterCollection(initialize, length(times), get_transform(annotation), get_prior(annotation))
        grid = variable ? Dict(t => i for (i, t) in enumerate(grid)) : grid
        push!(locals, (; variable, entry = (; index, sym = entry.sym, name = entry.name, grid, parameter, fixed = !is_tunable(annotation))))
    end
    return (;
        variables = [local_entry.entry for local_entry in locals if local_entry.variable],
        parameters = [local_entry.entry for local_entry in locals if !local_entry.variable],
    )
end

function update_experiment_bounds!(bounds, entries)
    for entry in entries
        isnothing(entry.parameter) || (bounds[entry.sym] = get_bounds(entry.parameter.transform))
    end
    return bounds
end

function merge_experiment_collection(global_layer, local_layer, variable)
    if variable
        children = merge(global_layer.initialconditions, local_layer.initialconditions)
        global_names, local_names = keys(global_layer.initialconditions), keys(local_layer.initialconditions)
    else
        children = merge(global_layer.parameters, local_layer.parameters)
        global_names, local_names = keys(global_layer.parameters), keys(local_layer.parameters)
    end
    indices = values(merge(NamedTuple{global_names}(global_layer.indices), NamedTuple{local_names}(local_layer.indices)))
    return variable ? InitialConditions(global_layer.name, global_layer.repack, children, indices) :
        Parameter(global_layer.name, global_layer.repack, children, indices)
end

"""
$(TYPEDEF)

A data-only Lux container owning shared and experiment-local parameters, controls,
and initial variables. Calling it returns per-experiment parameter/state tuples.
A separate ShootingLayer executes the experiments.

# Fields
$(FIELDS)
"""
@concrete terse struct Experiments <: LuxCore.AbstractLuxContainerLayer{(:shared, :experiments)}
    "Global initial variables, parameters, and controls, initialized once."
    shared
    "Internal local-value providers, with no shooting layers."
    experiments
    "Prepared per-experiment assignments, problems, and symbol metadata."
    configurations
    "The problem supplying symbols, shapes, and natural defaults; no solver is stored."
    problem
end

"""
    Experiments(problem, shared_definitions...; experiments, kwargs...)
    Experiments(shooting::ShootingLayer, experiments::NamedTuple)

Define shared and experiment-local parameters, controls, and initial variables.
No shooting layer or solver is stored. The problem supplies defaults and indexing;
shared definitions and metadata keywords follow `ShootingLayer` without its solver.
The second form inherits an existing layer's definitions during construction only.

Each named entry accepts `annotations`, `observations`, and `u0`. Local annotations
use `annotation => timegrid` syntax and may redefine any model parameter or variable
as `Tunable` or `Constant`, with its own grid, bounds, transform, and prior.
Undeclared quantities remain shared. Globals overridden everywhere are omitted
from setup, including their unused unknowns, priors, and initializers.

`LuxCore.setup` returns `ps.shared` and `ps.experiments.<name>`, with fixed values and
schedules in `st`. `experiments(nothing, ps, st)` returns `(calls, st)`, where each
entry of `calls` is a merged `(experiment_ps, experiment_st)` tuple. A separate
`shooting(experiments, ps, st)` returns a vector of trajectories and updated state,
in the named declaration order. ComponentArrays are supported.
"""
Experiments(shooting::ShootingLayer, experiments::NamedTuple) = build_experiments(shooting, experiments)

function build_experiments(shooting, experiments::NamedTuple)
    isempty(experiments) && throw(ArgumentError("Supply at least one experiment."))
    specs = map(experiments) do spec
        spec isa Experiment && return (; observations = spec)
        spec isa NamedTuple || throw(ArgumentError("Experiment entries must be NamedTuples or observation Experiments."))
        all(key -> key in (:annotations, :observations, :u0), keys(spec)) ||
            throw(ArgumentError("Experiment entries accept annotations, observations, and u0 only."))
        spec
    end
    global_variables = experiment_entries(shooting.initialcondition, shooting.sys, true)
    global_parameters = experiment_entries(shooting.parameter, shooting.sys, false)
    entries = map(spec -> local_experiment_entries(shooting, get(spec, :annotations, ()), global_variables, global_parameters), specs)
    # A quantity used by no experiment must not contribute an unknown or a prior.
    retained_variables = filter(global_variables) do entry
        any(local_entries -> all(local_entry -> local_entry.index != entry.index, local_entries.variables), entries)
    end
    retained_parameters = filter(global_parameters) do entry
        any(local_entries -> all(local_entry -> local_entry.index != entry.index, local_entries.parameters), entries)
    end
    variable_overrides = Set(entry.index for local_entries in entries for entry in local_entries.variables)
    parameter_overrides = Set(entry.index for local_entries in entries for entry in local_entries.parameters)
    # Retained globals that can be locally replaced use stable scalar keys. Purely
    # global groups and entirely local groups retain their shared-grid execution.
    shared = (;
        initialcondition = experiment_collection(shooting.initialcondition, retained_variables, InitialCondition; separate = variable_overrides),
        parameter = experiment_collection(shooting.parameter, retained_parameters, PiecewiseConstant; separate = parameter_overrides),
    )
    retained_variable_indices = Set(entry.index for entry in retained_variables)
    retained_parameter_indices = Set(entry.index for entry in retained_parameters)
    locals = map(specs, entries) do spec, local_entries
        data = get(spec, :observations, nothing)
        data = data isa Experiment || isnothing(data) ? data : Experiment(data...)
        u0 = get(spec, :u0, isnothing(data) ? nothing : data.u0)
        isnothing(u0) || length(u0) == length(shooting.problem.u0) ||
            throw(DimensionMismatch("Experiment u0 must match the state length."))
        if !isnothing(data)
            data = Experiment(data.observations, u0, data.t)
            all(t -> first(shooting.problem.tspan) <= t <= last(shooting.problem.tspan), data.t) ||
                throw(ArgumentError("Observation times must lie within the problem time span."))
        end
        initialcondition = experiment_collection(
            shooting.initialcondition, local_entries.variables, InitialCondition;
            separate = retained_variable_indices
        )
        parameter = experiment_collection(
            shooting.parameter, local_entries.parameters, PiecewiseConstant;
            separate = retained_parameter_indices
        )
        return (; layer = ExperimentLayer(initialcondition, parameter, data), u0)
    end
    # Explicit transforms can replace model bounds even on shared quantities.
    shared_bounds = Dict{Any, Any}(default_bounds(shooting.sys))
    update_experiment_bounds!(shared_bounds, global_variables)
    update_experiment_bounds!(shared_bounds, global_parameters)
    configurations = map(locals, entries) do local_entry, local_entries
        local_layer = local_entry.layer
        initialcondition = merge_experiment_collection(shared.initialcondition, local_layer.initialcondition, true)
        parameter = merge_experiment_collection(shared.parameter, local_layer.parameter, false)
        problem = isnothing(local_entry.u0) ? shooting.problem : remake(shooting.problem; u0 = local_entry.u0)
        bounds = copy(shared_bounds)
        update_experiment_bounds!(bounds, local_entries.variables)
        update_experiment_bounds!(bounds, local_entries.parameters)
        # Transform bounds are read directly from the owning Lux children.
        return (;
            initialcondition, parameter, problem, sys = ControlSymbolCache(shooting.sys; bounds),
            initial_permutation = collection_permutation(initialcondition.indices),
            parameter_permutation = collection_permutation(parameter.indices),
        )
    end
    return Experiments(shared, map(entry -> entry.layer, locals), configurations, shooting.problem)
end

function Experiments(
        problem::SciMLBase.AbstractSciMLProblem, definitions::Pair...;
        experiments::NamedTuple, fixed_parameters = [], controls = nothing, quadratures = nothing, bounds = nothing
    )
    definitions = isempty(definitions) ? Any[] : collect(definitions)
    ic_defs = findall(definition -> is_variable(first(definition)), definitions)
    p_defs = setdiff(eachindex(definitions), ic_defs)
    sys = shooting_metadata(problem, definitions, ic_defs; controls, quadratures, bounds)
    initialcondition = InitialConditions(problem, definitions[ic_defs]...; sys)
    parameter = Parameter(problem, definitions[p_defs]...; fixed = fixed_parameters, sys)
    return build_experiments((; problem, initialcondition, parameter, sys), experiments)
end

# Emit literal field access from the Lux state's NamedTuple schema. This also
# gives ComponentArray.getproperty a compile-time field name for every view.
@generated function experiment_fields(x, ::NamedTuple{names}) where {names}
    fields = [:(getproperty(x, $(QuoteNode(name)))) for name in names]
    return :(NamedTuple{$(QuoteNode(names))}(($(fields...),)))
end
function merge_experiment_ps(global_ps, local_ps, global_st::NamedTuple, local_st::NamedTuple)
    initials = merge(
        experiment_fields(global_ps.initialcondition.initialconditions, global_st.initialcondition.initialconditions),
        experiment_fields(local_ps.initialcondition.initialconditions, local_st.initialcondition.initialconditions)
    )
    parameters = merge(
        experiment_fields(global_ps.parameter.parameters, global_st.parameter.parameters),
        experiment_fields(local_ps.parameter.parameters, local_st.parameter.parameters)
    )
    return (; initialcondition = (; initialconditions = initials), parameter = (; parameters))
end

function merge_experiment_st(config, global_st, local_st, active)
    return (;
        initialcondition = (;
            initialconditions = merge(global_st.initialcondition.initialconditions, local_st.initialcondition.initialconditions),
            active, permutation = config.initial_permutation,
        ),
        parameter = (; parameters = merge(global_st.parameter.parameters, local_st.parameter.parameters), permutation = config.parameter_permutation),
    )
end

# This immutable execution view only combines prepared references. No annotation
# processing, setup, problem remake, or numerical buffer allocation happens here.
experiment_shooting(shooting::ShootingLayer, config) = ShootingLayer(
    shooting.name, config.problem, shooting.alg,
    shooting.ensemblealg, config.initialcondition, config.parameter, config.sys
)
function LuxCore.initialstates(rng::Random.AbstractRNG, layer::Experiments)
    shared = LuxCore.initialstates(rng, layer.shared)
    experiments = map(experiment -> LuxCore.initialstates(rng, experiment), layer.experiments)
    names = keys(layer.experiments)
    execution = NamedTuple{names}(
        map(names) do name
            config = layer.configurations[name]
            state = merge_experiment_st(config, shared, experiments[name], fill(false, sum(length, config.initialcondition.indices; init = 0)))
            times = shooting_timestops(config.initialcondition, config.parameter, config.problem, state, eltype(config.problem.tspan))
            data = layer.experiments[name].observations
            isnothing(data) || (times = sort!(unique!(vcat(times, data.t))))
            (; active = state.initialcondition.active, timestops = times)
        end
    )
    return (; shared, experiments, execution)
end

function prepare_experiment(layer, ::Val{name}, ps, st) where {name}
    experiment, config = getproperty(layer.experiments, name), getproperty(layer.configurations, name)
    local_ps, local_st = experiment(nothing, getproperty(ps.experiments, name), getproperty(st.experiments, name))
    merged_ps = merge_experiment_ps(ps.shared, local_ps, st.shared, local_st)
    execution = getproperty(st.execution, name)
    merged_st = merge(
        merge_experiment_st(config, st.shared, local_st, execution.active), (;
            timestops = execution.timestops,
            experiment_configuration = (; problem = layer.problem, configuration = config),
        )
    )
    return merged_ps, merged_st
end

# Generation inspects types and field names only, never numeric values or user
# initializers. Declaration order is retained even if state fields are reordered.
function experiment_state_names(layer_type, state_type)
    hasfield(state_type, :experiments) && hasfield(state_type, :execution) || return nothing
    declared = fieldnames(fieldtype(layer_type, :experiments))
    available = fieldnames(fieldtype(state_type, :experiments))
    execution = fieldnames(fieldtype(state_type, :execution))
    length(declared) == length(available) == length(execution) &&
        all(name -> name in available && name in execution, declared) || return nothing
    return declared
end

@generated function (layer::Experiments)(::Any, ps, st::NamedTuple)
    names = experiment_state_names(layer, st)
    isnothing(names) && return :(throw(ArgumentError("Experiment state fields must match the declared experiments.")))
    calls = [:(prepare_experiment(layer, Val($(QuoteNode(name))), ps, st)) for name in names]
    return :(($(calls...),), st)
end

# The returned tuples are directly executable. Their state carries only prepared
# annotation metadata; the caller supplies the solver through ShootingLayer.
function (shooting::ShootingLayer)(::Nothing, ps, st::NamedTuple)
    if hasproperty(st, :experiment_configuration)
        context = st.experiment_configuration
        shooting.problem === context.problem || throw(ArgumentError("ShootingLayer and experiment state must describe the same problem."))
        config = context.configuration
        execution = experiment_shooting(shooting, config)
        return execution(config.problem, ps, st)
    end
    return shooting(shooting.problem, ps, st)
end

function execute_experiment(shooting, layer, ::Val{name}, call, state) where {name}
    experiment = getproperty(layer.experiments, name)
    trajectory, next = shooting(nothing, call...)
    local_st = getproperty(state.experiments, name)
    local_initials = experiment_fields(next.initialcondition.initialconditions, local_st.initialcondition.initialconditions)
    local_parameters = experiment_fields(next.parameter.parameters, local_st.parameter.parameters)
    local_active = update_active_ics!(
        local_st.initialcondition.active,
        values(experiment.initialcondition.initialconditions), values(local_initials), local_st.initialcondition.permutation
    )
    local_next = (;
        initialcondition = merge(local_st.initialcondition, (; initialconditions = local_initials, active = local_active)),
        parameter = merge(local_st.parameter, (; parameters = local_parameters)),
    )
    return (; trajectory, state = local_next, execution = (; active = next.initialcondition.active, timestops = next.timestops))
end

# Keep a Vector API without widening mixed trajectories to abstract Trajectory.
# A finite union preserves each experiment's concrete result type.
function experiment_trajectory_vector(trajectories::T) where {T <: Tuple}
    element_type = Union{fieldtypes(T)...}
    return element_type[trajectories...]
end

@generated function (shooting::ShootingLayer)(layer::Experiments, ps, st::NamedTuple)
    names = experiment_state_names(layer, st)
    isnothing(names) && return :(throw(ArgumentError("Experiment state fields must match the declared experiments.")))
    results = [gensym(:experiment) for _ in names]
    evaluations = [:($(results[i]) = execute_experiment(shooting, layer, Val($(QuoteNode(name))), calls[$i], state)) for (i, name) in enumerate(names)]
    trajectories = [:(getproperty($result, :trajectory)) for result in results]
    states = [:(getproperty($result, :state)) for result in results]
    executions = [:(getproperty($result, :execution)) for result in results]
    return quote
        calls, state = layer(nothing, ps, st)
        $(evaluations...)
        trajectories = experiment_trajectory_vector(($(trajectories...),))
        next = merge(
            state, (;
                experiments = NamedTuple{$(QuoteNode(names))}(($(states...),)),
                execution = NamedTuple{$(QuoteNode(names))}(($(executions...),)),
            )
        )
        return trajectories, next
    end
end

function get_number_of_shooting_constraints(layer::Experiments, ps, st)
    return sum(keys(layer.experiments)) do name
        config = layer.configurations[name]
        state = merge(merge_experiment_st(config, st.shared, st.experiments[name], st.execution[name].active), (; timestops = st.execution[name].timestops))
        shooting_constraint_count(config.initialcondition, config.parameter, config.problem, state)
    end
end

# Keep the generic container bounds contract unchanged. Within experiment
# containers, scalar annotations supply the bounds, with no parallel metadata tree.
experiment_bounds(layer, ps, select) = unbounded_parameter_bounds(ps, select((-Inf, Inf)))
experiment_bounds(layer::ParameterCollection, ps, select) = parameter_bound_tree(ps, select(get_bounds(layer.transform)))
experiment_bounds(::FixedLayer, ps, select) = (;)
function experiment_bounds(layer::LuxCore.AbstractLuxContainerLayer{fields}, ps, select) where {fields}
    operation = (child, values, state) -> experiment_bounds(child, values, select)
    return container_parameter_bounds(operation, layer, ps, nothing, fields)
end
function experiment_bounds(layers::NamedTuple, ps, select)
    names = optimization_parameter_names(ps, keys(layers))
    return NamedTuple{names}(map(name -> experiment_bounds(layers[name], getproperty(ps, name), select), names))
end
get_lower_bound(layer::Experiments, ps, st = nothing) = experiment_bounds(layer, ps, first)
get_upper_bound(layer::Experiments, ps, st = nothing) = experiment_bounds(layer, ps, last)
