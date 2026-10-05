using Corleone, CorleoneParameterEstimation
using OrdinaryDiffEqTsit5
using NonlinearSolve
using LuxCore
using CairoMakie
using Test
using Random
using StableRNGs

rng = StableRNG(77)

# Lotka-Volterra-like model, two observed quantities.
f = (du, u, p, t) -> (
    du[1] = u[1] - p[1] * prod(u);
    du[2] = -u[2] + p[2] * prod(u);
    nothing
)
p_true = [1.0, 1.0]
prob = ODEProblem(f, [0.5, 0.7], (0.0, 5.0), [1.3, 0.8])
obs = (u, p, t) -> u[1:2]
t = collect(0.0:0.25:5.0)

# Generate synthetic data by simulating the truth.
true_layer = SingleShootingLayer(prob, Tsit5(); saveat = unique(t))
ps, st = LuxCore.setup(rng, true_layer)
ps = merge(ps, (p = p_true,))
traj, _ = true_layer(nothing, ps, st)
pe_data = ParameterEstimationLayer(
    true_layer, ExperimentData(t, zeros(2, length(t))); observed = obs)
Y = model_values(pe_data, traj, t)
data = ExperimentData(t, Y; σ = 0.05)

pe = ParameterEstimationLayer(prob, Tsit5(), data; observed = obs)
sol = solve(problem(pe), GaussNewton(); abstol = 1.0e-8, reltol = 1.0e-8)
result = fitted_parameters(pe, sol)

@testset "CairoMakie plotting extension: data" begin
    fig = plot(data)
    @test fig isa Figure
    fig2 = plot(data; idxs = [1], markersize = 3)
    @test fig2 isa Figure
end

@testset "CairoMakie plotting extension: fit overlay" begin
    fig = plot(result)
    @test fig isa Figure
    # overlay into an existing axis (lines + scatter for both quantities)
    ax = Axis(Figure()[1, 1])
    plot!(ax, result)
    plot!(ax, data)
    @test length(ax.elements) >= 2
    fig3 = plot(result; idxs = [1, 2], experiment_indices = [1], labels = ["prey", "pred"])
    @test fig3 isa Figure
end

@testset "CairoMakie plotting extension: multi-experiment" begin
    data2 = ExperimentData(
        Experiment(t, Y; σ = 0.05, u0 = [0.5, 0.7]),
        Experiment(t, Y; σ = 0.05, u0 = [0.4, 0.6]),
    )
    pe2 = ParameterEstimationLayer(prob, Tsit5(), data2; observed = obs)
    sol2 = solve(problem(pe2), GaussNewton(); abstol = 1.0e-8, reltol = 1.0e-8)
    result2 = fitted_parameters(pe2, sol2)
    @test result2.trajectory isa Tuple
    fig = plot(result2)
    @test fig isa Figure
    fig2 = plot(result2; experiment_indices = [2])
    @test fig2 isa Figure
end
