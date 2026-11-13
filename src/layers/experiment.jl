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


function local_experiment_entries(layer, definitions, variables, controls, observation_times)
    definitions = definitions isa Union{Pair, AbstractAnnotation} ? (definitions,) : definitions
    resolved = reduce(
        vcat, map(definitions) do definition
            if definition isa AbstractAnnotation && !is_variable(definition)
                isempty(observation_times) && throw(ArgumentError("Supply control switching times when the batch has no observation times."))
                definition = definition => filter(t -> t > first(layer.problem.tspan), observation_times)
            end
            definition isa Pair && first(definition) isa AbstractAnnotation ||
                throw(ArgumentError("Use a control annotation or an annotation => timegrid pair."))
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
        entries = variable ? variables : controls
        position = findfirst(entry -> entry.index == index, entries)
        isnothing(position) && throw(ArgumentError("Only controls can have experiment-local parameter annotations."))
        entry = entries[position]
        grid = variable && isnothing(grid) ? Float64[] : grid
        times = variable ? grid : isnothing(grid) ? [first(layer.problem.tspan)] : vcat(first(layer.problem.tspan), grid)
        default, transform = get_default(annotation), get_transform(annotation)
        initialize = let default = default, times = times, transform = transform
            (rng, n) -> initialize_annotation_values(default, rng, n, times, transform)
        end
        parameter = ParameterCollection(initialize, length(times), transform, get_prior(annotation))
        grid = variable ? Dict(t => i for (i, t) in enumerate(grid)) : grid
        push!(locals, (; variable, entry = (; index, sym = entry.sym, name = entry.name, grid, parameter, fixed = !is_tunable(annotation))))
    end
    parameters = [local_entry.entry for local_entry in locals if !local_entry.variable]
    length(parameters) == length(controls) ||
        throw(ArgumentError("Supply a Constant or Tunable annotation for every control in each experiment."))
    return (;
        variables = [local_entry.entry for local_entry in locals if local_entry.variable],
        parameters,
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

A data-only Lux container owning shared model parameters, experiment-local
controls, and shared or local initial variables. Calling it returns
per-experiment parameter/state tuples.
A separate ShootingLayer executes the experiments.

# Fields
$(FIELDS)
"""
# A heap-allocated container with constant, concretely typed fields: passing or
# capturing it moves a pointer instead of copying several kilobytes of inline
# layer data, while field access stays fully inferable.
mutable struct Experiments{S, E, C, P, D} <: LuxCore.AbstractLuxContainerLayer{(:shared, :experiments)}
    "Shared initial variables and model parameters, initialized once."
    const shared::S
    "Internal local-value providers, with no shooting layers."
    const experiments::E
    "Prepared per-experiment assignments, problems, and symbol metadata."
    const configurations::C
    "The problem supplying symbols, shapes, and natural defaults; no solver is stored."
    const problem::P
    "The original annotation layers and experiment specifications."
    const definition::D
end
# Match the terse type printing of the other layers.
Base.show(io::IO, T::Type{<:Experiments}) = print(io, T isa UnionAll ? "Experiments" : "Experiments{}")

"""
    Experiments(problem, shared_definitions...; experiments, kwargs...)
    Experiments(shooting::ShootingLayer, experiments::NamedTuple)

Share ordinary model parameters across experiments and configure controls and
initial variables for individual experiments.
No shooting layer or solver is stored. The problem supplies defaults and indexing;
shared definitions and metadata keywords follow `ShootingLayer` without its solver.
The second form inherits an existing layer's definitions during construction only.

Each named entry accepts `annotations`, `observations`, and `u0`. Each control
requires a local `Tunable` or `Constant` annotation. A bare control annotation uses
the union of observation times in the batch, with one initial value at the start
of the problem. Use `annotation => timegrid` to supply an explicit schedule or
`annotation => nothing` for a single value throughout the experiment. Initial
variables can also have local annotations with an explicit grid. Shared initial
variables overridden everywhere are omitted from setup.

`LuxCore.setup` returns `ps.shared` and `ps.experiments.<name>`, with fixed values and
schedules in `st`. Observation times from all experiments share `st.saveat`.
`experiments(nothing, ps, st)` returns `(calls, st)`, where each
entry of `calls` is a merged `(experiment_ps, experiment_st)` tuple. A separate
`shooting(experiments, ps, st)` returns a vector of trajectories and updated state,
in the named declaration order. ComponentArrays are supported.
"""
Experiments(shooting::ShootingLayer, experiments::NamedTuple) = build_experiments(shooting, experiments)

function build_experiments(shooting, experiments::NamedTuple; warm = nothing)
    isempty(experiments) && throw(ArgumentError("Supply at least one experiment."))
    specs = map(experiments) do spec
        spec isa Experiment && (spec = (; observations = spec))
        spec isa NamedTuple || throw(ArgumentError("Experiment entries must be NamedTuples or observation Experiments."))
        all(key -> key in (:annotations, :observations, :u0), keys(spec)) ||
            throw(ArgumentError("Experiment entries accept annotations, observations, and u0 only."))
        data = get(spec, :observations, nothing)
        data = data isa Experiment || isnothing(data) ? data : Experiment(data...)
        u0 = get(spec, :u0, isnothing(data) ? nothing : data.u0)
        isnothing(u0) || length(u0) == length(shooting.problem.u0) ||
            throw(DimensionMismatch("Experiment u0 must match the state length."))
        if !isnothing(data)
            data = Experiment(data.observations, u0, data.t)
            experiment_sample_times(shooting.problem, data)
        end
        return (; data, u0, annotations = get(spec, :annotations, ()))
    end
    global_variables = experiment_entries(shooting.initialcondition, shooting.sys, true)
    parameters = experiment_entries(shooting.parameter, shooting.sys, false)
    indices = control_indices(shooting.sys)
    controls = filter(entry -> entry.index in indices, parameters)
    shared_parameters = filter(entry -> entry.index ∉ indices, parameters)
    observation_times = experiment_observation_times(map(spec -> spec.data, values(specs)))
    entries = map(spec -> local_experiment_entries(shooting, spec.annotations, global_variables, controls, observation_times), specs)
    # A quantity used by no experiment must not contribute an unknown or a prior.
    retained_variables = filter(global_variables) do entry
        any(local_entries -> all(local_entry -> local_entry.index != entry.index, local_entries.variables), entries)
    end
    variable_overrides = Set(entry.index for local_entries in entries for entry in local_entries.variables)
    if !isnothing(warm)
        retained_variables = warm_experiment_entries(retained_variables, warm.shared.initialcondition, shooting.problem)
        shared_parameters = warm_experiment_entries(shared_parameters, warm.shared.parameter, shooting.problem)
    end
    # Retained globals that can be locally replaced use stable scalar keys. Purely
    # global groups and entirely local groups retain their shared-grid execution.
    shared = (;
        initialcondition = experiment_collection(shooting.initialcondition, retained_variables, InitialCondition; separate = variable_overrides),
        parameter = experiment_collection(shooting.parameter, shared_parameters, PiecewiseConstant; separate = ()),
    )
    retained_variable_indices = Set(entry.index for entry in retained_variables)
    locals = map(keys(specs), values(specs), values(entries)) do name, spec, local_entries
        if !isnothing(warm) && hasproperty(warm.experiments, name)
            previous = warm.experiments[name]
            local_entries = (;
                variables = warm_experiment_entries(local_entries.variables, previous.initialcondition, shooting.problem),
                parameters = warm_experiment_entries(local_entries.parameters, previous.parameter, shooting.problem),
            )
        end
        initialcondition = experiment_collection(
            shooting.initialcondition, local_entries.variables, InitialCondition;
            separate = retained_variable_indices
        )
        parameter = experiment_collection(
            shooting.parameter, local_entries.parameters, PiecewiseConstant;
            separate = ()
        )
        return (; layer = ExperimentLayer(initialcondition, parameter, spec.data), spec.u0)
    end
    # Explicit transforms can replace model bounds even on shared quantities.
    shared_bounds = Dict{Any, Any}(default_bounds(shooting.sys))
    update_experiment_bounds!(shared_bounds, global_variables)
    update_experiment_bounds!(shared_bounds, shared_parameters)
    locals = NamedTuple{keys(specs)}(locals)
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
    source = (; shooting.problem, shooting.initialcondition, shooting.parameter, shooting.sys)
    return Experiments(shared, map(entry -> entry.layer, locals), configurations, shooting.problem, (; source, experiments))
end

function experiment_observation_times(data)
    measured = filter(!isnothing, data)
    isempty(measured) && return Float64[]
    return sort!(unique!(copy(reduce(vcat, map(experiment -> experiment.t, measured)))))
end

warm_branch(layer, ps, st) = (layer, ps, st)
warm_branch(layer::FixedLayer, ps, st) = (layer.layer, st.parameters, st.states)

function experiment_warm_values(layer, ps, st)
    variable = layer isa InitialConditions
    children = variable ? layer.initialconditions : layer.parameters
    parameters = variable ? ps.initialconditions : ps.parameters
    states = variable ? st.initialconditions : st.parameters
    result = Dict{Int, Any}()
    for (name, indices) in zip(keys(children), layer.indices)
        child = children[name]
        child isa PassThrough && continue
        child, child_ps, child_st = warm_branch(child, getproperty(parameters, name), states[name])
        collections = child.parameter
        if collections isa NamedTuple
            for (index, key) in zip(indices, keys(collections))
                collection, values, _ = warm_branch(
                    collections[key], getproperty(child_ps.parameter, key), child_st.parameter[key]
                )
                result[index] = (; grid = child_st.lookup.collection, values = natural_values(collection.transform, values.ps))
            end
        else
            result[only(indices)] = (; grid = child_st.lookup.collection, values = natural_values(collections.transform, child_ps.parameter.ps))
        end
    end
    return result
end

function experiment_warm_values(layer::Experiments, ps, st)
    names = keys(layer.experiments)
    local_values = map(names) do name
        experiment_warm_values(layer.experiments[name], getproperty(ps.experiments, name), st.experiments[name])
    end
    return (;
        shared = experiment_warm_values(layer.shared, ps.shared, st.shared),
        experiments = NamedTuple{names}(local_values),
    )
end

experiment_warm_values(layer::Union{NamedTuple, ExperimentLayer}, ps, st) = (;
    initialcondition = experiment_warm_values(layer.initialcondition, ps.initialcondition, st.initialcondition),
    parameter = experiment_warm_values(layer.parameter, ps.parameter, st.parameter),
)

function warm_experiment_entries(entries, previous, problem)
    return map(entries) do entry
        haskey(previous, entry.index) && !isnothing(entry.parameter) || return entry
        old = previous[entry.index]
        values = if isequal(old.grid, entry.grid)
            copy(old.values)
        elseif entry.grid isa AbstractDict
            all(t -> haskey(old.grid, t), keys(entry.grid)) || return entry
            [old.values[old.grid[t]] for t in sort!(collect(keys(entry.grid)); by = t -> entry.grid[t])]
        else
            times = isnothing(entry.grid) ? [first(problem.tspan)] : vcat(first(problem.tspan), entry.grid)
            [old.values[lookup!(nothing, old.grid, t)] for t in times]
        end
        collection = entry.parameter
        return merge(entry, (; parameter = ParameterCollection((rng, n) -> copy(values), collection.out_dims, collection.transform, collection.prior)))
    end
end

"""
    append_experiment(layer::Experiments, ps, st, experiment::Pair; rng = Random.default_rng())

Return `(; layer, ps, st)` for a new batch with one appended named experiment.
The pair's value follows the `Experiments` constructor. Reuse existing values by
experiment name and model index. When an inferred control grid expands, seed
each interval from the previous piecewise-constant control at its start.
"""
function append_experiment(layer::Experiments, ps, st, (name, spec)::Pair{Symbol}; rng = Random.default_rng())
    definitions = layer.definition.experiments
    haskey(definitions, name) && throw(ArgumentError("Experiment `$name` already exists."))
    warm = experiment_warm_values(layer, ps, st)
    next = build_experiments(layer.definition.source, merge(definitions, NamedTuple{(name,)}((spec,))); warm)
    parameters, state = LuxCore.setup(rng, next)
    return (; layer = next, ps = parameters, st = state)
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
            (; active = state.initialcondition.active, timestops = times)
        end
    )
    data = filter(!isnothing, map(experiment -> experiment.observations, values(layer.experiments)))
    saveat = if isempty(data)
        nothing
    else
        get_saveat(layer.problem, reduce(vcat, map(experiment -> experiment.t, data)), map(entry -> entry.timestops, values(execution)))
    end
    return (; shared, experiments, execution, saveat)
end

get_saveat(::Experiments, st) = st.saveat

prepare_experiment(layer, ::Val{name}, ps, st) where {name} = prepare_experiment(
    getproperty(layer.experiments, name), getproperty(layer.configurations, name), layer.problem,
    ps.shared, getproperty(ps.experiments, name), st.shared, getproperty(st.experiments, name),
    getproperty(st.execution, name), get_saveat(layer, st)
)

# Experiments are selected by the caller, so this barrier sees only their values:
# experiments with equal types share one compiled method, and the unrolled
# per-experiment body stays a short list of calls.
@noinline function prepare_experiment(experiment, config, problem, shared_ps, experiment_ps, shared_st, experiment_st, execution, saveat)
    local_ps, local_st = experiment(nothing, experiment_ps, experiment_st)
    merged_ps = merge_experiment_ps(shared_ps, local_ps, shared_st, local_st)
    merged_st = merge(
        merge_experiment_st(config, shared_st, local_st, execution.active), (;
            timestops = execution.timestops, saveat,
            experiment_configuration = (; problem, configuration = config),
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

execute_experiment(shooting, layer, ::Val{name}, call, state) where {name} = execute_experiment(
    shooting, getproperty(layer.experiments, name), call, getproperty(state.experiments, name), getproperty(state.execution, name)
)

@noinline function execute_experiment(shooting, experiment, call, local_st, experiment_execution)
    trajectory, next = shooting(nothing, call...)
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
    execution = merge(experiment_execution, (; active = next.initialcondition.active))
    return (; trajectory, state = local_next, execution)
end

# Keep a Vector API without widening mixed trajectories to abstract Trajectory.
# A finite union preserves each experiment's concrete result type.
function experiment_trajectory_vector(trajectories::T) where {T <: Tuple}
    element_type = Union{fieldtypes(T)...}
    return element_type[trajectories...]
end

function (shooting::ShootingLayer)(layer::Experiments, ps, st::NamedTuple)
    trajectories, next = experiment_trajectories(shooting, layer, ps, st)
    return experiment_trajectory_vector(trajectories), next
end

# Internal form of the experiment forward pass: one trajectory per declared
# experiment as a Tuple, so callers that know the experiment names keep each
# trajectory's concrete type (a Vector would lose the length, a Union the type).
@generated function experiment_trajectories(shooting::ShootingLayer, layer::Experiments, ps, st::NamedTuple)
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
        trajectories = ($(trajectories...),)
        next = merge(
            state, (;
                experiments = NamedTuple{$(QuoteNode(names))}(($(states...),)),
                execution = NamedTuple{$(QuoteNode(names))}(($(executions...),)),
            )
        )
        return trajectories, next
    end
end

# Map over the NamedTuples (reordered to declaration order) rather than indexing
# them with runtime Symbols, so every experiment's count stays inferable.
function get_number_of_shooting_constraints(layer::Experiments, ps, st)
    names = keys(layer.configurations)
    counts = map(layer.configurations, NamedTuple{names}(st.experiments), NamedTuple{names}(st.execution)) do config, local_st, execution
        state = merge(merge_experiment_st(config, st.shared, local_st, execution.active), (; timestops = execution.timestops))
        shooting_constraint_count(config.initialcondition, config.parameter, config.problem, state)
    end
    return sum(values(counts); init = 0)
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
# Experiment parameters are continuous estimation variables.
get_integrality(layer::Experiments, ps, st = nothing) = experiment_bounds(layer, ps, _ -> false)
