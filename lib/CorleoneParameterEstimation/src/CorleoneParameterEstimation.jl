module CorleoneParameterEstimation

using Reexport
@reexport using Corleone
using DocStringExtensions
using LuxCore
using Random
using LinearAlgebra
using SparseArrays
using SciMLBase
using SciMLStructures
using SymbolicIndexingInterface
using ForwardDiff

using NonlinearSolve
using NonlinearSolve: NonlinearLeastSquaresProblem

# Re-export the least-squares solver algorithms so users get the standard
# SciML GN/LM interface without an extra `using NonlinearSolve`.
@reexport using NonlinearSolve: GaussNewton, LevenbergMarquardt,
    BoundedLevenbergMarquardt, TrustRegion, Dogbox, TrustRegionDogleg, NewtonRaphson

include("data.jl")
export Experiment, ExperimentData, experiments, n_experiments, n_observations,
    observation_times, initial_conditions, controls, has_controls

include("layer.jl")
export ParameterEstimationLayer

include("residuals.jl")
export residuals!, residual, model_values

include("problem.jl")
export CorleoneParameterEstimationProblem, problem

include("uncertainty.jl")
export ParameterEstimate, ParameterEstimationResult
export estimate_parameters, covariance, standard_errors, confidence_interval, fitted_parameters

end
