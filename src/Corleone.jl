module Corleone

using Reexport
using DocStringExtensions
import PrecompileTools: @compile_workload, @setup_workload

import LinearAlgebra
using SparseArrays
using Random
using ConcreteStructs

using SciMLBase
using SciMLPublic: @public
using SciMLStructures
using SymbolicIndexingInterface
import RecursiveArrayTools

using OhMyThreads
using Distributed

using ChainRulesCore

using LuxCore

"""
    trajectoryplot(trajectory; figure = (;), kwargs...)

Create a Makie figure displaying trajectory states, controls, and quadratures.
Load Makie to activate this method. `figure` configures the figure; remaining
keywords are passed to [`trajectoryplot!`](@ref).
"""
function trajectoryplot end
"""
    trajectoryplot!(container, trajectory; skips = [], show_legend = true, kwargs...)

Add trajectory axes and optional legends to a Makie figure or layout position.
`skips` excludes selected symbols. State, control, and quadrature axes share the
time axis. Load Makie to activate this method; remaining keywords configure lines.
"""
function trajectoryplot! end

export trajectoryplot, trajectoryplot!

# For evaluation
mythreadmap(::EnsembleSerial, args...) = map(args...)
mythreadmap(::EnsembleThreads, args...) = tmap(args...)
mythreadmap(::EnsembleDistributed, args...) = pmap(args...)

# Extensions
is_extension_loaded(::Val) = false

include("optimization.jl")
include("utilities/Utilities.jl")

include("layers/Layers.jl")
export ShootingLayer

include("utilities/parameter_collections.jl")

include("utilities/experiments.jl")
include("layers/experiment.jl")
export Experiments, append_experiment
export Observation, Experiment, experiment_state, residual, n_observations

export get_bounds, get_lower_bound, get_upper_bound, get_integrality, get_number_of_shooting_constraints
export shooting_constraints, shooting_constraints!, add_parameter_segments!, adapt_timegrid!
export Stages

include("optimization_expressions.jl")
export default_evaluator, compile_evaluator
include("layers/evaluation.jl")
export EvaluationLayer
include("dynprob.jl")
export CorleoneDynamicOptProblem, optimization_objective, optimization_constraints, optimization_constraints!
include("optimization_bridge.jl")
export pack, default_vectorizer, constraint_jacobian_sparsity, lagrangian_hessian_sparsity
# Supported hooks for sublibraries that wrap shooting layers; see the API reference.
@public evaluation_parameters, compose_evaluation, compile_observation_expression, experiment_components, experiment_trajectories
@public optimization_dependency_rows, dependency_column_ids, shooting_dependency_context
@public trajectory_constraint_dependencies, expression_signals, dependency_pattern

include("deprecation/controls.jl")
include("deprecation/single_shooting.jl")
include("multiple_shooting.jl")
export ControlParameter, SingleShootingLayer, MultipleShootingLayer

include("multistage.jl")
export StageTransition, StageLinkage, linkage_residual, get_number_of_linkage_constraints

include("precompile.jl")

end
