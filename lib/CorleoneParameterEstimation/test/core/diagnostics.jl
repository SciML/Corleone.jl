using Corleone, CorleoneParameterEstimation
using OrdinaryDiffEqTsit5
using NonlinearSolve
using LuxCore
using LinearAlgebra: diag
using Test
using Random
using StableRNGs

rng = StableRNG(2024)

# Model: du/dt = -p[1]*u, exact solution u(t) = u0 * exp(-p[1]*t). The initial condition is
# made tunable so that there are two fitted parameters (u0, p) and the correlation matrix and
# condition number are non-trivial.
f = (du, u, p, t) -> (du[1] = -p[1] * u[1]; nothing)
p_true = 0.7
u0_true = 1.0
prob = ODEProblem(f, [u0_true], (0.0, 5.0), [0.3])

tdata = collect(0.0:0.25:5.0)
u_exact = u0_true .* exp.(-p_true .* tdata)
σ = 0.01
ydata = u_exact .+ σ .* randn(rng, length(tdata))
data = ExperimentData(tdata, ydata; σ = σ)

pe = ParameterEstimationLayer(prob, Tsit5(), data;
    observed = (u, p, t) -> u[1:1], tunable_ic = [1])
sol = solve(problem(pe), GaussNewton(); abstol = 1.0e-10, reltol = 1.0e-10)
est = estimate_parameters(pe, sol)

@testset "Goodness-of-fit diagnostics" begin
    @test SciMLBase.successful_retcode(sol)
    # A near-noiseless fit should have R² close to one.
    @test r_squared(est) > 0.99
    @test adjusted_r_squared(est) <= r_squared(est)
    @test adjusted_r_squared(est) > 0.99
    @test est.r_squared == r_squared(est)
    @test est.adjusted_r_squared == adjusted_r_squared(est)
end

@testset "Correlation and identifiability diagnostics" begin
    C = correlation(est)
    @test size(C) == (est.n_parameters, est.n_parameters)
    @test all(isapprox.(diag(C), 1.0; atol = 1.0e-12))
    @test C ≈ C' atol = 1.0e-12
    @test all(abs.(C) .<= 1.0 + 1.0e-12)
    # The covariance's diagonal is reproduced by the correlation and standard errors.
    @test C[1, 2] ≈ est.covariance[1, 2] / (est.std_errors[1] * est.std_errors[2])
    # A positive, finite condition number is reported.
    @test isfinite(condition_number(est))
    @test condition_number(est) >= 1.0
    @test condition_number(est) == est.condition_number
end

@testset "Safe inversion of a singular information matrix" begin
    # A rank-deficient (exactly singular) matrix must not abort; the pseudo-inverse fallback
    # is used instead and a warning is emitted.
    A = [1.0 1.0; 1.0 1.0]
    invA = @test_logs (:warn, r"singular or numerically rank-deficient") match_mode = :any begin
        CorleoneParameterEstimation._safe_inverse(A)
    end
    @test invA ≈ [0.25 0.25; 0.25 0.25] atol = 1.0e-12
end

@testset "ParameterEstimate printing respects the confidence level" begin
    s95 = sprint(show, est)
    @test occursin("95% CI", s95)
    est90 = estimate_parameters(pe, sol; α = 0.1)
    s90 = sprint(show, est90)
    @test occursin("90% CI", s90)
    @test !occursin("95% CI", s90)
end
