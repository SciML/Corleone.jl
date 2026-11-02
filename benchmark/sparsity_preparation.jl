# Run separately from tests and other benchmarks. See benchmark/README.md.
using Corleone, ComponentArrays, OrdinaryDiffEqTsit5, Statistics

function report(name, f; samples = 20)
    f()
    f()
    measurements = map(1:samples) do _
        sample = @timed f()
        # Retain scalar measurements, rather than all generated sparse matrices.
        (; time = sample.time, bytes = sample.bytes)
    end
    println(name, ",seconds=", median(x.time for x in measurements), ",bytes=", median(x.bytes for x in measurements))
    return flush(stdout)
end

function pattern_case(n, points; reads = :none)
    grid = collect(range(0.05, 1.95; length = points))
    prob = ODEProblem((u, p, t) -> p, ones(n), (0.0, 2.0), ones(n))
    layer = ShootingLayer(prob, Tsit5(), Tunable(collect(1:n); kind = Val(:variable), default = ones(n)) => grid, Tunable(collect(1:n); default = ones(n)) => grid)
    if reads === :none
        return CorleoneDynamicOptProblem(layer, traj -> sum(last(traj.u)))
    end
    syms = Corleone.parameter_symbols(layer.sys)
    expressions = reads === :few ? [Expr(:call, Symbol(first(syms)), t) for t in grid] : [Expr(:call, Symbol(sym), 1.0) for sym in syms]
    return CorleoneDynamicOptProblem(layer, traj -> sum(last(traj.u)), Expr(:vect, expressions...); lcons = fill(-Inf, length(expressions)), ucons = fill(Inf, length(expressions)))
end
for (name, n, points, reads) in (("pattern_small", 24, 19, :none), ("pattern_large", 96, 39, :none), ("few_parameter_reads", 128, 19, :few), ("all_parameter_reads", 128, 19, :all))
    model = pattern_case(n, points; reads)
    report(name, () -> constraint_jacobian_sparsity(model))
end

# Dense solver saving should not require dependency snapshots at every sample.
prob = ODEProblem((u, p, t) -> p, ones(3), (0.0, 2.0), ones(3); saveat = collect(range(0.0, 2.0; length = 10001)))
layer = ShootingLayer(prob, Tsit5(), Tunable([1, 2, 3]; kind = Val(:variable), default = ones(3)) => [0.75, 1.25], Tunable([1, 2, 3]; default = ones(3)) => [0.5, 1.0, 1.5])
model = CorleoneDynamicOptProblem(layer, :(u₁(2.0)), :([sum(u₁), sum(p₁), u₂(1.0), p₂(0.5)]); lcons = zeros(4), ucons = fill(Inf, 4))
report("history_10001_samples", () -> constraint_jacobian_sparsity(model))
