# Include problem definitions once, before callers can obtain their constructors.
# Keeping discovery here preserves automatic registration of new problem files.
const BENCHMARK_FILES = filter(name -> endswith(name, ".jl"), readdir(joinpath(@__DIR__, "problems")))
for file in BENCHMARK_FILES
    include(joinpath("problems", file))
end
const BENCHMARK_CONSTRUCTORS = map(BENCHMARK_FILES) do file
    name = Symbol(splitext(file)[1])
    constructor = getfield(@__MODULE__, name)
    constructor isa Function || throw(ArgumentError("$file must define a benchmark constructor named $name."))
    constructor
end

"""
$(SIGNATURES)

Return the benchmark constructor functions in alphabetical order. Definitions are
loaded once with the package; repeated calls do not redefine methods. Each
constructor accepts [`BenchmarkGrids`](@ref) and returns a system, independent
scaled grids, declared state/control counts, time span, and descriptive name.
"""
load_benchmarks() = copy(BENCHMARK_CONSTRUCTORS)
