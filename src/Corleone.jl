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

"""
    experiments_from_dataframe(df; observables = nothing, initial_states = nothing)

Import PEtab 2-style measurement rows into a named tuple of [`Experiment`](@ref)s,
keyed by `experimentId` in first-appearance order. Requires DataFrames.
Columns are `observableId`, `experimentId`, `time`, and `measurement`.
`observables` maps IDs (string or symbol keys) to state selectors, callbacks, or
expressions; by default IDs become state symbols. `initial_states` maps experiment
IDs to `u0` vectors. Repeated measurements are retained as independent residuals.
An optional `noiseParameters` column accepts positive numeric standard deviations
or numeric strings; blank entries are unweighted. Full PEtab problem import,
parameterized noise/observable overrides, preequilibration, and infinite times
are unsupported and rejected.
"""
function experiments_from_dataframe end

"""
    trajectories_dataframe(trajectory; experiment_id = "experiment1", observables = nothing)
    trajectories_dataframe(trajectories; experiment_ids = nothing, observables = nothing)

Export saved trajectories as a PEtab 2-style DataFrame with `observableId`,
`experimentId`, `time`, and `simulation` columns. Requires DataFrames.
`observables` maps output IDs to state symbols, indices, or scalar callable objects
`(u, p, t) -> value`; the default exports every state. Collections may be vectors,
tuples, or named tuples. Named tuple keys supply experiment IDs; otherwise IDs
are `experiment1`, `experiment2`, … unless `experiment_ids` is supplied.
Extra columns `sampleIndex` (one-based saved sample index) and `resetSide`
(`"incoming"`, `"outgoing"`, or `"none"`) preserve both sides of resets.
`DataFrame(trajectory; kwargs...)` and `DataFrame(trajectory_vector; kwargs...)`
are shorthand for the corresponding forms.
"""
function trajectories_dataframe end

export experiments_from_dataframe, trajectories_dataframe

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
