module OptimalControlBenchmarks

using ModelingToolkit

include("types.jl")
include("plot_solution.jl")
include("problem_registry.jl")
include("solver_corleone.jl")
include("run_benchmarks.jl")

export load_benchmarks, run_all

end