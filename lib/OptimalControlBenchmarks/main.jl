using OptimalControlBenchmarks
using Corleone
using UnoSolver
using Ipopt
using ModelingToolkit

benchmarks = load_benchmarks()

# choose the optimizer
# optimizer = UnoSolver.Optimizer()
optimizer = Ipopt.Optimizer()

# settings for discretization, given as grids in [0.,1.]

# the constraint grid is assumed to contain 0. and 1.
constraint_grid = collect(0.:0.05:1.)
control_grid = collect(0.:0.05:1.)[1:end-1]
shooting_grid = collect(0.:0.05:1.)

grids = OptimalControlBenchmarks.BenchmarkGrids(
	constraint_grid,
	control_grid,
	shooting_grid
)

# run all benchmarks from the `problems` folder
df = run_all(benchmarks, optimizer, grids)

println(df)
