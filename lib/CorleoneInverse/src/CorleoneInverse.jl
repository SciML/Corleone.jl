module CorleoneInverse

using Reexport: @reexport
@reexport using Corleone
using Corleone: Corleone, BoundedTransform, ShootingLayer, Experiments, CorleoneDynamicOptProblem,
    ExperimentDynamicOptProblem, ObservationExperiments, default_vectorizer, is_weighted,
    get_bounds, log_prior, natural_values, n_observations, optimization_constraints, pack, parameter_collections, residual
using ADTypes: AbstractADType
using ConcreteStructs: @concrete
using DocStringExtensions: FIELDS, SIGNATURES, TYPEDEF
using Distributions: Normal, TDist, mean, quantile, std
using ForwardDiff: ForwardDiff
using LinearAlgebra: I, Symmetric, diag, nullspace
using SciMLBase: SciMLBase

include("problem.jl")
include("statistics.jl")
export ParameterEstimate

end
