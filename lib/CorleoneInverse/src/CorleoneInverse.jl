module CorleoneInverse

using Reexport: @reexport
@reexport using Corleone
import Corleone: CorleoneDynamicOptProblem, evaluate_trajectory_objective,
    get_number_of_shooting_constraints, n_observations, optimization_matching,
    optimization_matching!, optimization_prediction, optimization_schedule, residual
using Corleone: Corleone, BoundedTransform, ShootingLayer, Experiment, Experiments, compile_evaluator,
    default_evaluator, default_vectorizer, get_bounds, log_prior, natural_values, optimization_constraints,
    pack, parameter_collections, shooting_constraints,
    collect_expression_times!, compile_trajectory_expression, experiment_fields, expression_sample_grid,
    expression_symbol_indices, expression_symbol_name, get_saveat, observation_residual, optimization_times,
    optimization_trajectory, prepare_constraint_bounds, prepare_dynamic_opt_problem, replace_expression_samples,
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
using SymbolicIndexingInterface: independent_variable_symbols

include("experiment_problem.jl")
include("problem.jl")
include("statistics.jl")
export ParameterEstimate, mean_squared_residual

include("precompile.jl")

end
