module OptimalControlBenchmarks

using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using ModelingToolkit: inputs
using OptimalControlBenchmarks
using CairoMakie
using Corleone
using OrdinaryDiffEqTsit5
using Optimization
using OptimizationMOI
using ForwardDiff
using ComponentArrays
using LuxCore, Random
using Symbolics

include("types.jl")
include("plot_solution.jl")
include("problem_registry.jl")
include("run_benchmarks.jl")
include("scale_grid.jl")
include("solver_corleone.jl")

export load_benchmarks, run_all

end
