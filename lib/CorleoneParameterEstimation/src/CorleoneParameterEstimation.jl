module CorleoneParameterEstimation

using Reexport
@reexport using Corleone
using DocStringExtensions
using LuxCore
using Random
using LinearAlgebra
using SciMLBase
using SciMLStructures
using SymbolicIndexingInterface
using ForwardDiff

using NonlinearSolve
using NonlinearSolve: NonlinearLeastSquaresProblem
using StatsFuns: tdistinvcdf

include("data.jl")
export Experiment, ExperimentData, experiments, n_experiments, n_observations,
    observation_times, observation_times_per_quantity, per_quantity,
    initial_conditions, controls, has_controls

include("layer.jl")
export ParameterEstimationLayer

include("residuals.jl")
export residuals!, residual, model_values, n_shooting_constraints

include("problem.jl")
export CorleoneParameterEstimationProblem, problem

include("uncertainty.jl")
export ParameterEstimate, ParameterEstimationResult
export estimate_parameters, covariance, standard_errors, confidence_interval, fitted_parameters
export r_squared, adjusted_r_squared, correlation, condition_number

end
