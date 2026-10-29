module OptimalControlBenchmarks

using ModelingToolkit
using DocStringExtensions
using ModelingToolkitBase: t_nounits as t, D_nounits as D
using ModelingToolkitBase: inputs
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

export BenchmarkGrids, load_benchmarks, prepare_benchmark, solve_with_corleone, run_all

end
