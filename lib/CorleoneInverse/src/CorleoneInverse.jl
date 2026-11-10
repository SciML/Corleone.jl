module CorleoneInverse

using Reexport: @reexport
@reexport using Corleone
using Corleone: Corleone, BoundedTransform, ShootingLayer, Experiment, Experiments, default_vectorizer, get_bounds, get_number_of_shooting_constraints,
    ObservationExperiments, as_experiments, experiment_state, is_weighted, log_prior, natural_values, n_observations,
    pack, parameter_collections, residual, shooting_constraints
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
