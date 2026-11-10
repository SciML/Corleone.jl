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
    :CorleoneDynamicOptProblem, :optimization_objective, :optimization_constraints,
    :optimization_constraints!, :pack, :default_vectorizer, :default_evaluator, :compile_evaluator,
    :constraint_jacobian_sparsity, :lagrangian_hessian_sparsity,
    :forward_initialization, :linear_initialization, :random_initialization,
    :constant_initialization, :custom_initialization, :hybrid_initialization,
    :trajectoryplot, :trajectoryplot!, :log_prior,
    :BoundedTransform, :Observation, :Experiment, :experiment_state, :residual, :n_observations,
)
run_qa(
    CorleoneInverse;
    # This sublibrary intentionally adds SciML problem constructors for its parent
    # package's shooting layers; these adapters are part of the shared API.
    aqua_kwargs = (; piracies = (; treat_as_own = [ShootingLayer])),
    ei_kwargs = (;
        # Established AD and symbolic indexing APIs that are not declared public.
        all_qualified_accesses_are_public = (; ignore = (:jacobian, :hessian)),
        # Corleone's parameter-collection hooks for transforms and priors.
        all_explicit_imports_are_public = (; ignore = (:Experiments, :as_experiments, :is_weighted, :natural_values, :parameter_collections)),
    ),
    api_docs_kwargs = (; docs_src = joinpath(@__DIR__, "..", "..", "docs", "src")),
    reexports_allow = CORLEONE_REEXPORTS,
)
