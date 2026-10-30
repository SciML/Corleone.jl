"""
$(TYPEDEF)

Description of one benchmark problem registered by `load_benchmarks`.

# Fields
$(FIELDS)

This is a developer type. Benchmark packages should provide a constructor with the
same callable contract as `make_problem` rather than depending on implementation
details of the benchmark registry.
"""
struct OptimalControlBenchmark{F}
    "Stable identifier used in benchmark output."
    name::Symbol
    "Human-readable description of the problem."
    description::String
    "Callable accepting `BenchmarkGrids` and returning benchmark problem data."
    make_problem::F
end

"""
$(TYPEDEF)

The grids used by an optimal-control benchmark constructor.

# Fields
$(FIELDS)

# Usage contract
User grids contain sorted, unique normalized time points in `[0, 1]`. Constructors
return independent grids scaled to their problem horizon. A
benchmark constructor accepts a `BenchmarkGrids` value and returns the problem data
consumed by `run_all`.
"""
struct BenchmarkGrids
    "Control interval starts; a final endpoint is optional and discarded during preparation."
    control_grid::Vector{Float64}
    "Shooting boundaries; an empty grid selects single shooting."
    shooting_grid::Vector{Float64}
    "Points at which path or terminal constraints are evaluated."
    constraint_grid::Vector{Float64}
end

BenchmarkGrids(; control_grid, shooting_grid = Float64[], constraint_grid = [0.0, 1.0]) =
    BenchmarkGrids(collect(Float64, control_grid), collect(Float64, shooting_grid), collect(Float64, constraint_grid))

(benchmark::OptimalControlBenchmark)(grids::BenchmarkGrids) = benchmark.make_problem(grids)
