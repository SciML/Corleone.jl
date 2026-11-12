module CorleoneInverse

using Reexport: @reexport
@reexport using Corleone
import Corleone: CorleoneDynamicOptProblem, append_experiment, evaluate_trajectory_objective,
    get_number_of_shooting_constraints, n_observations, optimization_matching,
    optimization_matching!, optimization_prediction, optimization_schedule, residual, log_prior, parameter_collections
using Corleone: Corleone, BoundedTransform, ShootingLayer, Experiment, Experiments, EvaluationLayer,
    default_evaluator, default_vectorizer, get_bounds, natural_values, optimization_constraints,
    pack, shooting_constraints,
    collect_expression_times!, compile_observation_expression, compile_trajectory_expression, evaluation_parameter_tree, evaluation_parameters,
    experiment_fields, expression_sample_grid,
    expression_symbol_indices, get_saveat, observation_residual, observation_symbol_indices, optimization_times,
    optimization_trajectory, prepare_constraint_bounds, prepare_dynamic_opt_problem,
    sample_indices, scalar_objective, state_index
using ADTypes: AbstractADType, AutoForwardDiff
using ComponentArrays: ComponentArray
using ConcreteStructs: @concrete
using DocStringExtensions: FIELDS, SIGNATURES, TYPEDEF
using Distributions: Normal, TDist, mean, quantile, std
using ForwardDiff: ForwardDiff
using LinearAlgebra: I, Symmetric, diag, nullspace
using LuxCore: LuxCore
using NonlinearSolve: LevenbergMarquardt
using OptimizationOptimisers: Adam
using OrdinaryDiffEqTsit5: Tsit5
using PrecompileTools: @setup_workload, @compile_workload
using Random: Random
using SciMLBase: SciMLBase
using SymbolicIndexingInterface: symbolic_container

include("experiment_problem.jl")
include("problem.jl")
include("statistics.jl")
export ParameterEstimate, mean_squared_residual

include("precompile.jl")

end
