using Corleone, CorleoneParameterEstimation
using OrdinaryDiffEqTsit5
using NonlinearSolve
using LuxCore
using SymbolicIndexingInterface
using SciMLBase
using Test
using Random
using StableRNGs

# Model: three decoupled decays, du[k]/dt = -p[k] u[k]. p[1], p[2] and p[3] are all
# tunable by default; the `params` keyword restricts which of them are actually estimated,
# the remaining parameters staying fixed at their value in `prob.p`.
f = (du, u, p, t) -> begin
    du[1] = -p[1] * u[1]
    du[2] = -p[2] * u[2]
    du[3] = -p[3] * u[3]
    nothing
end
p_true = [0.7, 0.5, 0.9]
prob = ODEProblem(f, [1.0, 1.0, 1.0], (0.0, 5.0), p_true)
tt = collect(0.0:0.5:5.0)
obs = (u, p, t) -> u[1:3]
σ = 0.05
rng = StableRNG(11)

# Exact (noise-free) trajectory used to build data for the recovery tests.
pe_true = ParameterEstimationLayer(
    prob, Tsit5(), ExperimentData(
        Experiment(tt, zeros(3, length(tt)); σ = σ, u0 = [1.0, 1.0, 1.0])
    ); observed = obs
)
ps_true, st_true = LuxCore.setup(rng, first(pe_true.layer))
ps_true = merge(ps_true, (p = p_true,))
traj_true, _ = first(pe_true.layer)(nothing, ps_true, st_true)
M = model_values(pe_true, traj_true, tt)

@testset "params reduces the optimization variables" begin
    # All three parameters tunable by default.
    pe_all = ParameterEstimationLayer(
        prob, Tsit5(), ExperimentData(
            Experiment(tt, M .+ σ .* randn(rng, size(M)); σ = σ, u0 = [1.0, 1.0, 1.0])
        );
        observed = obs
    )
    nlls_all = problem(pe_all)
    # u0 is fixed (default), p[1], p[2], p[3] tunable.
    @test length(nlls_all.u0) == 3

    # Estimate only p[1] and p[3]; p[2] fixed at its value in prob.p.
    pe_sub = ParameterEstimationLayer(
        prob, Tsit5(), ExperimentData(
            Experiment(tt, M .+ σ .* randn(rng, size(M)); σ = σ, u0 = [1.0, 1.0, 1.0])
        );
        observed = obs, params = [1, 3]
    )
    nlls_sub = problem(pe_sub)
    @test length(nlls_sub.u0) == 2
    # The two free entries correspond to p[1] and p[3] (initialized from prob.p).
    @test nlls_sub.u0 == [p_true[1], p_true[3]]

    # Selecting a single parameter leaves exactly one free parameter.
    pe_one = ParameterEstimationLayer(
        prob, Tsit5(), ExperimentData(
            Experiment(tt, M; σ = σ, u0 = [1.0, 1.0, 1.0])
        ); observed = obs, params = [2]
    )
    @test length(problem(pe_one).u0) == 1

    # Selection by parameter Symbol names is equivalent to selection by index. Attach a
    # `SymbolCache` so the parameters carry names `k1`, `k2`, `k3`.
    fnamed = ODEFunction(f; sys = SymbolCache([:u1, :u2, :u3], [:k1, :k2, :k3], :t))
    prob_named = ODEProblem(fnamed, [1.0, 1.0, 1.0], (0.0, 5.0), p_true)
    pe_sym = ParameterEstimationLayer(
        prob_named, Tsit5(), ExperimentData(
            Experiment(tt, M; σ = σ, u0 = [1.0, 1.0, 1.0])
        ); observed = obs,
        params = [:k1, :k3]
    )
    @test length(problem(pe_sym).u0) == 2
end

@testset "params keeps unselected parameters fixed" begin
    # A noiseless fit of p[1], p[3] must leave p[2] untouched at its prob.p value.
    y = M
    pe = ParameterEstimationLayer(
        prob, Tsit5(), ExperimentData(
            Experiment(tt, y; σ = σ, u0 = [1.0, 1.0, 1.0])
        ); observed = obs, params = [1, 3]
    )
    nlls = problem(pe)
    sol = solve(nlls, GaussNewton(); abstol = 1.0e-8, reltol = 1.0e-8, maxiters = 500)
    @test sol.u ≈ [p_true[1], p_true[3]] atol = 1.0e-6

    # Reconstruct the full parameter vector from the estimate: p[2] stays at its fixed value
    # (prob.p[2]) while p[1] and p[3] are the fitted entries.
    est = estimate_parameters(pe, sol)
    @test length(est.std_errors) == 2

    # A noiseless fit restricted to p[2] only recovers p[2] and leaves p[1], p[3] fixed.
    pe2 = ParameterEstimationLayer(
        prob, Tsit5(), ExperimentData(
            Experiment(tt, y; σ = σ, u0 = [1.0, 1.0, 1.0])
        ); observed = obs, params = [2]
    )
    sol2 = solve(problem(pe2), GaussNewton(); abstol = 1.0e-8, reltol = 1.0e-8, maxiters = 500)
    @test sol2.u ≈ [p_true[2]] atol = 1.0e-6
end

@testset "params with multiple experiments" begin
    # Same model, two experiments sampled at different times; both share the tunable subset.
    w1 = collect(0.0:0.5:5.0)
    w2 = collect(0.5:0.5:5.0)
    y1 = M[:, indexin(w1, tt)] .+ σ .* randn(rng, 3, length(w1))
    y2 = M[:, indexin(w2, tt)] .+ σ .* randn(rng, 3, length(w2))
    data = ExperimentData(
        Experiment(w1, y1; σ = σ, u0 = [1.0, 1.0, 1.0]),
        Experiment(w2, y2; σ = σ, u0 = [1.0, 1.0, 1.0]),
    )
    pe = ParameterEstimationLayer(prob, Tsit5(), data; observed = obs, params = [1, 3])
    nlls = problem(pe)
    @test length(nlls.u0) == 2
    sol = solve(nlls, GaussNewton(); abstol = 1.0e-8, reltol = 1.0e-8, maxiters = 500)
    @test sol.u ≈ [p_true[1], p_true[3]] atol = 3σ
end

@testset "params with multiple shooting" begin
    shooting = [1.0, 3.0]
    msdata = ExperimentData(
        Experiment(tt, M .+ σ .* randn(rng, size(M)); σ = σ, u0 = [1.0, 1.0, 1.0])
    )
    pe = ParameterEstimationLayer(
        prob, Tsit5(), msdata, shooting;
        observed = obs, params = [1, 3]
    )
    nlls = problem(pe)
    # The tunable p block (p[1], p[3]) is [p_true[1], p_true[3]]; shooting initial
    # conditions add further variables.
    @test nlls.u0[1:2] == [p_true[1], p_true[3]]
    @test length(nlls.u0) > 2
    sol = solve(nlls, GaussNewton(); abstol = 1.0e-8, reltol = 1.0e-8, maxiters = 500)
    @test sol.u[1:2] ≈ [p_true[1], p_true[3]] atol = 3σ
end

@testset "params validation" begin
    # Control slots cannot be selected as tunable parameters.
    ctrl = 1 => ControlParameter(collect(0.0:1.0:5.0), controls = ones(6))
    @test_throws ArgumentError ParameterEstimationLayer(
        prob, Tsit5(), ExperimentData(
            Experiment(tt, M; σ = σ, u0 = [1.0, 1.0, 1.0])
        );
        observed = obs, controls = (ctrl,), params = [1, 3]
    )
    # Out-of-range index.
    @test_throws ArgumentError ParameterEstimationLayer(
        prob, Tsit5(), ExperimentData(
            Experiment(tt, M; σ = σ, u0 = [1.0, 1.0, 1.0])
        );
        observed = obs, params = [1, 99]
    )
    # Unknown parameter name.
    @test_throws ArgumentError ParameterEstimationLayer(
        prob, Tsit5(), ExperimentData(
            Experiment(tt, M; σ = σ, u0 = [1.0, 1.0, 1.0])
        );
        observed = obs, params = [:does_not_exist]
    )
end

@testset "params respects parameter transforms" begin
    # Transforms apply per tunable entry, so only the selected parameters are transformed.
    # Use noiseless data so the recovery is exact (isolating the transform mechanics).
    pe = ParameterEstimationLayer(
        prob, Tsit5(), ExperimentData(
            Experiment(tt, M; σ = σ, u0 = [1.0, 1.0, 1.0])
        );
        observed = obs, params = [1, 3]
    )
    nlls = problem(pe; transforms = [Log(), Log()])
    @test length(nlls.u0) == 2
    sol = solve(nlls, GaussNewton(); abstol = 1.0e-8, reltol = 1.0e-8, maxiters = 500)
    @test exp.(sol.u) ≈ [p_true[1], p_true[3]] atol = 1.0e-4
end
