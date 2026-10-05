using SciMLTesting
using CorleoneParameterEstimation
using Test

# These are the names `CorleoneParameterEstimation` exposes to users: its own documented
# API plus the deliberate `@reexport using` facades (`@reexport using Corleone` and
# `@reexport using NonlinearSolve: <solvers>`). Additions are user-facing API changes and
# must be reviewed with their documentation.
const CORLEONE_REEXPORTS = (
    :ControlParameter, :Corleone, :CorleoneDynamicOptProblem, :MultipleShootingLayer,
    :SingleShootingLayer, :Trajectory, :constant_initialization, :custom_initialization,
    :default_initialization, :forward_initialization, :hybrid_initialization,
    :linear_initialization, :random_initialization,
)
const SOLVERS_REEXPORTS = (
    :BoundedLevenbergMarquardt, :Dogbox, :GaussNewton, :LevenbergMarquardt,
    :NewtonRaphson, :TrustRegion, :TrustRegionDogleg,
)
const OWN_EXPORTS = (
    :CorleoneParameterEstimationProblem, :Experiment, :ExperimentData, :ParameterEstimate,
    :ParameterEstimationLayer, :ParameterEstimationResult, :adjusted_r_squared,
    :condition_number, :confidence_interval, :controls, :correlation, :covariance,
    :estimate_parameters, :experiments, :fitted_parameters,
    :has_controls, :initial_conditions, :model_values, :n_experiments, :n_observations,
    :n_shooting_constraints,
    :observation_times, :observation_times_per_quantity, :per_quantity, :problem, :r_squared, :residual, :residuals!, :standard_errors,
)
run_qa(
    CorleoneParameterEstimation;
    # The `@reexport using Corleone` facade pulls a large implicit-import surface (Corleone,
    # SymbolicIndexingInterface, Symbolics, ...), so the implicit-imports analysis cannot be
    # satisfied; this mirrors CorleoneOED (see SciML/Corleone.jl#103).
    ei_broken = (:no_implicit_imports,),
    ei_kwargs = (;
        # Names still not declared public in their owning modules, reached through
        # `Corleone.*`/`SciMLBase.*`/`ForwardDiff.*`: Corleone internals (`control_length`,
        # `get_bounds`, `get_number_of_shooting_constraints`, `get_problem`,
        # `shooting_constraints!`), SciMLBase internals (`get_colorizers`) and ForwardDiff's
        # `jacobian`.
        all_qualified_accesses_are_public = (;
            ignore = (
                :control_length, :get_bounds, :get_colorizers, :get_problem, :jacobian,
                :get_number_of_shooting_constraints, :shooting_constraints!,
            ),
        ),
    ),
    reexports_allow = (CORLEONE_REEXPORTS..., SOLVERS_REEXPORTS..., OWN_EXPORTS...),
)
