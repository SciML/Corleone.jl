using OptimalControlBenchmarks
using Corleone
using ModelingToolkit

benchmarks = load_benchmarks()

df = run_all(benchmarks)

println(df)