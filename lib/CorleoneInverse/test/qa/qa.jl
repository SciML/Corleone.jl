using SciMLTesting
using CorleoneInverse
using Test

# Corleone is re-exported as the user-facing facade; its own API is documented there.
const CORLEONE_REEXPORTS = (
    :Corleone, :ControlParameter, :SingleShootingLayer, :MultipleShootingLayer,
    :ShootingLayer, :Trajectory, :InitialConditions, :Parameter,
    :Annotations, :Tunable, :Constant, :ControlSymbolCache,
    :control_symbols, :control_indices, :quadrature_symbols, :quadrature_indices, :default_bounds,
    :get_bounds, :get_lower_bound, :get_upper_bound, :get_number_of_shooting_constraints,
    :shooting_constraints, :shooting_constraints!, :add_parameter_segments!,
    :CorleoneDynamicOptProblem, :EvaluationLayer, :optimization_objective, :optimization_constraints,
    :optimization_constraints!, :pack, :default_vectorizer, :default_evaluator, :compile_evaluator,
    :constraint_jacobian_sparsity, :lagrangian_hessian_sparsity,
    :forward_initialization, :linear_initialization, :random_initialization,
    :constant_initialization, :custom_initialization, :hybrid_initialization,
    :trajectoryplot, :trajectoryplot!, :log_prior,
    :BoundedTransform, :Experiments, :append_experiment, :Observation, :Experiment, :experiment_state, :residual, :n_observations,
    :StageLinkage, :StageTransition, :Stages, :get_number_of_linkage_constraints, :linkage_residual,
)
run_qa(
    CorleoneInverse;
    # This sublibrary intentionally adds SciML problem constructors for its parent
    # package's shooting layers; these adapters are part of the shared API.
    aqua_kwargs = (; piracies = (; treat_as_own = [ShootingLayer, CorleoneDynamicOptProblem])),
    ei_kwargs = (;
        # Established AD and symbolic indexing APIs that are not declared public.
        all_qualified_accesses_are_public = (; ignore = (:jacobian, :hessian)),
        # Corleone's dynamic optimization and observation preparation hooks.
        all_explicit_imports_are_public = (;
            ignore = (
                :collect_expression_times!, :compile_observation_expression, :compile_trajectory_expression,
                :evaluate_trajectory_objective, :evaluation_parameter_tree, :evaluation_parameters,
                :evaluation_collections, :evaluation_coordinate_ids, :evaluation_log_prior, :evaluation_parameter_metadata,
                :experiment_fields, :expression_sample_grid, :expression_symbol_indices,
                :get_saveat, :natural_values, :observation_residual, :observation_symbol_indices,
                :optimization_matching, :optimization_matching!,
                :optimization_prediction, :optimization_schedule, :optimization_times, :optimization_trajectory,
                :parameter_collections, :prepare_constraint_bounds,
                :prepare_dynamic_opt_problem, :sample_indices, :scalar_objective, :state_index,
                :check_optimization_schedule, :dependency_pattern, :experiment_shooting,
                :ObservationExpression, :ParsedTrajectoryExpression, :prepare_experiment,
                :optimization_dependency_rows, :shooting_dependency_context, :sparsity_indices,
                :trajectory_constraint_dependencies,
            ),
        ),
    ),
    api_docs_kwargs = (; docs_src = joinpath(@__DIR__, "..", "..", "docs", "src")),
    reexports_allow = CORLEONE_REEXPORTS,
)
