using Corleone, CorleoneParameterEstimation
using OrdinaryDiffEqTsit5
using NonlinearSolve
using LuxCore
using Test
using Random
using StableRNGs

# Model: two decoupled exponential decays, du[k]/dt = -p[k] u[k]. Both states are
# observed; each may be sampled at its own time points. p[1], p[2] are tunable.
f = (du, u, p, t) -> begin
    du[1] = -p[1] * u[1]
    du[2] = -p[2] * u[2]
    nothing
end
p_true = 0.7
prob = ODEProblem(f, [1.0, 2.0], (0.0, 5.0), [p_true, p_true])
obs = (u, p, t) -> u[1:2]
σ = 0.05
rng = StableRNG(7)

# Generate an "exact" trajectory on a fine grid to sample per-quantity data from. The grid
# uses a 0.5 step so that all sampled time points (also multiples of 0.5) match exactly.
tt = collect(0.0:0.5:5.0)
pe_true = ParameterEstimationLayer(
    prob, Tsit5(), ExperimentData(
        Experiment(tt, zeros(2, length(tt)); σ = σ, u0 = [1.0, 2.0])
    ); observed = obs
)
ps, st = LuxCore.setup(rng, first(pe_true.layer))
ps = merge(ps, (p = [p_true, p_true],))
traj, _ = first(pe_true.layer)(nothing, ps, st)
M = model_values(pe_true, traj, tt)

@testset "Per-quantity observation times: construction & accessors" begin
    # Two quantities, each with its own time grid and values.
    w1 = collect(0.0:1.0:5.0)
    w2 = collect(0.5:1.0:4.5)
    y1 = M[1, indexin(w1, tt)] .+ σ .* randn(rng, length(w1))
    y2 = M[2, indexin(w2, tt)] .+ σ .* randn(rng, length(w2))
    exp = Experiment([w1, w2], [y1, y2]; σ = σ, u0 = [1.0, 2.0])

    data = ExperimentData(exp)
    @test per_quantity(data, 1)
    @test n_observations(data) == length(w1) + length(w2)
    @test observation_times_per_quantity(data, 1) == [w1, w2]
    # the stored `t` is the (sorted) union grid, used to sample the model
    @test observation_times(data, 1) == sort(unique(vcat(w1, w2)))

    # The shared-grid matrix form is NOT per-quantity.
    data_shared = ExperimentData(Experiment(w1, ones(2, length(w1)); σ = σ, u0 = [1.0, 2.0]))
    @test !per_quantity(data_shared, 1)
    @test isnothing(observation_times_per_quantity(data_shared, 1))
end

@testset "Per-quantity observation times: validation" begin
    # Number of time vectors must match number of value vectors.
    @test_throws ArgumentError Experiment([collect(0.0:1.0:2.0), collect(0.0:1.0:2.0)], [ones(3)])
    # Per-quantity lengths must match.
    @test_throws ArgumentError Experiment(
        [collect(0.0:1.0:2.0), collect(0.0:1.0:2.0)],
        [ones(3), ones(4)]
    )
    # At least one quantity is required.
    @test_throws ArgumentError Experiment(AbstractVector{Float64}[], AbstractVector{Float64}[])
end

@testset "Per-quantity fit recovers parameters" begin
    w1 = collect(0.0:1.0:5.0)
    w2 = collect(0.5:1.0:4.5)
    y1 = M[1, indexin(w1, tt)] .+ σ .* randn(rng, length(w1))
    y2 = M[2, indexin(w2, tt)] .+ σ .* randn(rng, length(w2))
    data = ExperimentData(Experiment([w1, w2], [y1, y2]; σ = σ, u0 = [1.0, 2.0]))

    pe = ParameterEstimationLayer(prob, Tsit5(), data; observed = obs)
    nlls = problem(pe)
    # u0 is fixed (default), so only p is free (both p[1], p[2] tunable).
    @test length(nlls.u0) == 2
    sol = solve(nlls, GaussNewton(); abstol = 1.0e-8, reltol = 1.0e-8, maxiters = 500)
    # Gauss--Newton on noisy data may stop without the strict `Success` retcode; what matters
    # is that both parameters are recovered close to `p_true = 0.7`.
    @test all(abs.(sol.u .- p_true) .< 3σ)
    est = estimate_parameters(pe, sol)
    @test est.n_observations == length(w1) + length(w2)
    @test length(est.std_errors) == 2
end

@testset "Shared and per-quantity forms give the same fit" begin
    # The same underlying observation set can be expressed either as a shared-grid matrix
    # (both quantities at the union grid) or as per-quantity series sampled at the same
    # points; both must produce (essentially) the same estimate.
    w1 = collect(0.0:1.0:5.0)
    w2 = collect(0.5:1.0:4.5)
    u_union = sort(unique(vcat(w1, w2)))
    y1 = M[1, indexin(w1, tt)] .+ σ .* randn(rng, length(w1))
    y2 = M[2, indexin(w2, tt)] .+ σ .* randn(rng, length(w2))

    # shared form: fill the full union matrix, marking unobserved entries as not contributing
    # is what the matrix form cannot express, so build a matrix that observes everything by
    # evaluating the model at ALL union points instead (equivalent to w1 == w2 when they are
    # identical). For a strict equivalence test use identical per-quantity grids:
    w_eq1 = collect(0.0:0.5:5.0)
    w_eq2 = collect(0.0:0.5:5.0)
    eq1 = M[1, indexin(w_eq1, tt)] .+ σ .* randn(rng, length(w_eq1))
    eq2 = M[2, indexin(w_eq2, tt)] .+ σ .* randn(rng, length(w_eq2))

    data_shared = ExperimentData(
        Experiment(
            w_eq1, vcat(reshape(eq1, 1, :), reshape(eq2, 1, :));
            σ = σ, u0 = [1.0, 2.0]
        )
    )
    data_pq = ExperimentData(Experiment([w_eq1, w_eq2], [eq1, eq2]; σ = σ, u0 = [1.0, 2.0]))
    @test n_observations(data_shared) == n_observations(data_pq)

    pe_shared = ParameterEstimationLayer(prob, Tsit5(), data_shared; observed = obs)
    pe_pq = ParameterEstimationLayer(prob, Tsit5(), data_pq; observed = obs)
    s1 = solve(problem(pe_shared), GaussNewton(); abstol = 1.0e-8, reltol = 1.0e-8, maxiters = 500)
    s2 = solve(problem(pe_pq), GaussNewton(); abstol = 1.0e-8, reltol = 1.0e-8, maxiters = 500)
    @test s1.u ≈ s2.u atol = 1.0e-6
    @test all(abs.(s1.u .- p_true) .< 3σ)
end

@testset "Per-quantity times with mixed experiments" begin
    # One per-quantity experiment and one shared-grid experiment in the same fit.
    w1 = collect(0.0:1.0:5.0)
    w2 = collect(0.5:1.0:4.5)
    y1 = M[1, indexin(w1, tt)] .+ σ .* randn(rng, length(w1))
    y2 = M[2, indexin(w2, tt)] .+ σ .* randn(rng, length(w2))
    t3 = collect(0.0:0.5:5.0)
    y3 = M[:, indexin(t3, tt)] .+ σ .* randn(rng, size(M)[1], length(t3))

    exp_pq = Experiment([w1, w2], [y1, y2]; σ = σ, u0 = [1.0, 2.0])
    exp_shared = Experiment(t3, y3; σ = σ, u0 = [1.0, 2.0])
    data = ExperimentData(exp_pq, exp_shared)
    @test n_observations(data) == length(w1) + length(w2) + length(t3) * 2
    @test per_quantity(data, 1)
    @test !per_quantity(data, 2)

    pe = ParameterEstimationLayer(prob, Tsit5(), data; observed = obs)
    sol = solve(problem(pe), GaussNewton(); abstol = 1.0e-8, reltol = 1.0e-8, maxiters = 500)
    @test all(abs.(sol.u .- p_true) .< 3σ)
end
