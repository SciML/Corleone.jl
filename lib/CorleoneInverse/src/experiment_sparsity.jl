"""
    residual_jacobian_sparsity(model; vectorize = default_vectorizer(), ps = model.ps, st = model.st)

Return the Boolean Jacobian pattern of `residual(model, ps, st)` for a prepared
ODE experiment fit. Rows follow residual order and columns follow `pack(vectorize, ps)`.
The pattern excludes prior and constraint rows. Rebuild it after changing the
packing, schedule, saved grid, or observation layout.
"""
function residual_jacobian_sparsity(
        model::ExperimentDynamicOptProblem; vectorize = default_vectorizer(), ps = model.ps, st = model.st
    )
    initial, reconstruct = pack(vectorize, ps)
    ncolumns = length(initial)
    ids = experiment_sparsity_ids(model, ncolumns, reconstruct, st)
    rows = experiment_dependency_rows(model, ids, st; observations = true).residuals
    return dependency_pattern(rows, n_observations(model), ncolumns)
end

function experiment_sparsity_ids(model, ncolumns, reconstruct, st)
    check_optimization_schedule(model.source_timestops, optimization_schedule(model.layer, st))
    return sparsity_indices(reconstruct(collect(1:ncolumns)))
end

function experiment_dependency_inputs(owner::Experiments, prediction, ids, st)
    return map(keys(prediction.data)) do key
        parameters, state = prepare_experiment(owner, Val(key), ids, st)
        shooting = experiment_shooting(prediction.shooting, owner.configurations[key])
        (; shooting, parameters, state)
    end
end

function experiment_dependency_inputs(owner::ShootingLayer, prediction, ids, st)
    return map(_ -> (; shooting = prediction.shooting, parameters = ids, state = st), keys(prediction.data))
end

constraint_experiment(f) = nothing
constraint_experiment(::ExperimentConstraint{K}) where {K} = K
constraint_expression(f) = f
constraint_expression(f::ExperimentConstraint) = f.f

function experiment_dependency_rows(model, packed_ids, st; observations = false, hessian = false)
    prediction = model.layer
    owner = prediction.layer
    ids = experiment_parameters(prediction, packed_ids)
    evaluation = evaluation_parameters(prediction.evaluation, packed_ids)
    inputs = experiment_dependency_inputs(owner, prediction, ids, st)
    constraints = model.constraints
    resolvers = Dict{Union{Symbol, Int}, Any}()
    experiment_columns = Dict{Union{Symbol, Int}, BitSet}()
    matching = BitSet[]
    matching_hessian = BitSet[]
    residuals = BitSet[]
    for (key, input) in zip(keys(prediction.data), inputs)
        (; shooting, parameters, state) = input
        times = model.times[key]
        inspected = Any[constraint_expression(c.f) for c in constraints if constraint_experiment(c.f) == key]
        dependencies, resolve = shooting_dependency_context(
            shooting, parameters, state, times, evaluation, inspected; hessian, saved_samples = observations
        )
        resolvers[key] = resolve
        append!(matching, dependencies.matching)
        append!(matching_hessian, dependencies.matching_hessian)
        if any(c -> constraint_experiment(c.f) == key && !(constraint_expression(c.f) isa ParsedTrajectoryExpression), constraints)
            columns = BitSet()
            for (_, values, _) in parameter_collections(shooting, parameters, state)
                union!(columns, values.ps)
            end
            experiment_columns[key] = columns
        end
        observations || continue
        for observation in model.objective.data[key].observations
            signals = observation_signals(observation.selector, shooting)
            append!(residuals, (resolve(signals, index) for index in observation.indices))
        end
    end
    length(matching) == model.nshooting || throw(ArgumentError("Shooting layout changed; reconstruct CorleoneDynamicOptProblem."))
    return (; residuals, matching, matching_hessian, resolvers, experiment_columns)
end

observation_signals(selector::Integer, shooting) = ((:states, selector, nothing),)
observation_signals(selector::ObservationExpression, shooting) = selector.dependencies

# The callback receives the state and parameter vectors at one sample.
function observation_signals(selector, shooting)
    return [(kind, index, nothing) for (kind, index) in values(expression_symbol_indices(shooting))]
end

function optimization_dependency_rows(model::ExperimentDynamicOptProblem, initial, reconstruct, st; hessian = false)
    ncolumns = length(initial)
    ids = experiment_sparsity_ids(model, ncolumns, reconstruct, st)
    rows = experiment_dependency_rows(model, ids, st; observations = hessian, hessian)
    user = BitSet[]
    for constraint in model.constraints
        key = constraint_experiment(constraint.f)
        f = constraint_expression(constraint.f)
        if isnothing(key)
            append!(user, fill(BitSet(1:ncolumns), constraint.n))
        elseif f isa ParsedTrajectoryExpression
            append!(user, trajectory_constraint_dependencies(f, constraint.n, rows.resolvers[key], ncolumns, length(model.times[key])))
        else
            append!(user, fill(rows.experiment_columns[key], constraint.n))
        end
    end
    objective = hessian ? experiment_objective_cliques(model, rows.residuals, ids, st) : BitSet[]
    return (; user, rows.matching, rows.matching_hessian, objective)
end

function experiment_objective_cliques(model, residuals, ids, st)
    reducer = model.objective.reducer
    cliques = reducer === mean || reducer === sum ? residuals : [foldl(union!, residuals; init = BitSet())]
    # Annotation priors and their transforms act on individual coordinates.
    for (collection, values, inserted) in parameter_collections(model.layer, ids, st)
        isnothing(collection.prior) && continue
        for i in eachindex(values.ps)
            i in inserted || push!(cliques, BitSet((values.ps[i],)))
        end
    end
    return cliques
end
