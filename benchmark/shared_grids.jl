using Corleone, LuxCore, Random, SciMLBase, OrdinaryDiffEqTsit5, BenchmarkTools, ComponentArrays

const N = 24
problem = ODEProblem((u, p, t) -> p, zeros(N), (0.0, 2.0), ones(N))
grid = collect(0.1:0.1:1.9)
initials = [Tunable(i; kind = Val(:variable), default = 1.0) => grid for i in 1:N]
controls = [Tunable(i; default = 1.0) => grid for i in 1:N]
layer = ShootingLayer(problem, Tsit5(), initials..., controls...)
ps, st = LuxCore.setup(MersenneTwister(1), layer)
state = zeros(N)
layer.initialcondition((state, 0.5), ps.initialcondition, st.initialcondition)
layer.parameter(0.5, ps.parameter, st.parameter)
trajectory, _ = layer(nothing, ps, st)
@assert all(value -> isapprox(value, 1.1; atol = 1.0e-8), last(trajectory.u))

function report(name, trial)
    estimate = median(trial)
    return println(name, " ns=", estimate.time, " bytes=", estimate.memory, " allocations=", estimate.allocs)
end
report("initials", @benchmark $(layer.initialcondition)(($state, 0.5), $(ps.initialcondition), $(st.initialcondition)) samples = 1000 evals = 1)
report("parameters", @benchmark $(layer.parameter)(0.5, $(ps.parameter), $(st.parameter)) samples = 1000 evals = 1)
report("forward", @benchmark $layer(nothing, $ps, $st) samples = 300 evals = 1)
packed = ComponentArray(ps)
@assert first(layer(nothing, packed, st)).u ≈ trajectory.u
report("packed_forward", @benchmark $layer(nothing, $packed, $st) samples = 300 evals = 1)
