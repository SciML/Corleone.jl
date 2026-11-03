using SciMLTesting
using CorleoneParameterEstimation
using Test

# These are the names `CorleoneParameterEstimation` exposes to users: its own documented
# API plus the deliberate `@reexport using` facades (`@reexport using Corleone` and
# `@reexport using NonlinearSolve: <solvers>`). Additions are user-facing API changes and
# must be reviewed with their documentation.
const CORLEONE_REEXPORTS = (
    :Annotations, :Constant, :ControlParameter, :ControlSymbolCache, :Corleone,
    :CorleoneDynamicOptProblem, :InitialConditions, :MultipleShootingLayer, :Parameter,
    :ShootingLayer, :SingleShootingLayer, :Trajectory, :Tunable, :add_parameter_segments!,
    :compile_evaluator, :constant_initialization, :constraint_jacobian_sparsity,
    :control_indices, :control_symbols, :custom_initialization, :default_bounds,
    :default_evaluator, :default_vectorizer, :forward_initialization, :get_bounds,
    :get_lower_bound, :get_number_of_shooting_constraints, :get_upper_bound,
    :hybrid_initialization, :lagrangian_hessian_sparsity, :linear_initialization,
    :optimization_constraints, :optimization_constraints!, :optimization_objective, :pack,
    :quadrature_indices, :quadrature_symbols, :random_initialization, :shooting_constraints,
    :shooting_constraints!, :trajectoryplot, :trajectoryplot!,
)
const SOLVERS_REEXPORTS = (
    :BoundedLevenbergMarquardt, :Dogbox, :GaussNewton, :LevenbergMarquardt,
    :NewtonRaphson, :TrustRegion, :TrustRegionDogleg,
)
const OWN_EXPORTS = (
    :CorleoneParameterEstimationProblem, :Experiment, :ExperimentData,
    :AbstractParameterTransform, :AbstractRegularizer, :Affine, :Identity, :Log, :Logit,
    :L1Regularizer, :L2Regularizer, :ElasticNetRegularizer,
    :ParameterEstimate, :ParameterEstimationLayer, :ParameterEstimationResult,
    :adjusted_r_squared, :condition_number, :confidence_interval, :controls, :correlation,
    :covariance, :estimate_parameters, :experiments, :fitted_parameters, :has_controls,
    :initial_conditions, :model_values, :n_experiments, :n_observations,
    :n_shooting_constraints, :observation_times, :observation_times_per_quantity,
    :per_quantity, :problem, :r_squared, :residual, :residuals!, :standard_errors,
    :ProfileCurve, :ProfileLikelihoodEstimate, :profile_likelihood, :profiles,
)
run_qa(
    CorleoneParameterEstimation;
    # The `@reexport using Corleone` facade pulls a large implicit-import surface (Corleone,
    # SymbolicIndexingInterface, Symbolics, ...), so the implicit-imports analysis cannot be
    # satisfied; this mirrors CorleoneOED (see SciML/Corleone.jl#103).
    ei_broken = (:no_implicit_imports,),
    ei_kwargs = (;
        # Names still not declared public in their owning modules, reached through
        # `Corleone.*`/`SciMLBase.*`/`ForwardDiff.*`: Corleone internals (`AbstractAnnotation`,
        # `FixedLayer`, `InitialCondition`, `PiecewiseConstant`, `get_symbolic_container`,
        # `get_timestops`, `legacy_bounds`, `legacy_lift_states`), SymbolicIndexingInterface's
        # `parameter_values_at_time`, and the existing Corleone/SciMLBase/ForwardDiff internals
        # (`control_length`, `get_bounds`, `get_colorizers`, `get_problem`, `jacobian`,
        # `get_number_of_shooting_constraints`, `shooting_constraints!`).
        all_qualified_accesses_are_public = (;
            ignore = (
                :AbstractAnnotation, :FixedLayer, :InitialCondition, :PiecewiseConstant,
                :get_symbolic_container, :get_timestops, :legacy_bounds, :legacy_lift_states,
                :parameter_values_at_time,
                :control_length, :get_bounds, :get_colorizers, :get_problem, :jacobian,
                :get_number_of_shooting_constraints, :shooting_constraints!,
            ),
        ),
    ),
    api_docs_kwargs = (; docs_src = joinpath(@__DIR__, "..", "..", "docs", "src")),
    reexports_allow = (CORLEONE_REEXPORTS..., SOLVERS_REEXPORTS..., OWN_EXPORTS...),
)
