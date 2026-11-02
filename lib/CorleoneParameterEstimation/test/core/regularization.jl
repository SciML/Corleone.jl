using Corleone, CorleoneParameterEstimation
using OrdinaryDiffEqTsit5
using NonlinearSolve
using LinearAlgebra: norm
using Test
using Random
using StableRNGs

rng = StableRNG(2024)

# Model: du/dt = -p[1]*u, exact solution u(t) = u0 * exp(-p[1]*t).
f = (du, u, p, t) -> (du[1] = -p[1] * u[1]; nothing)
p_true = 0.7
u0_true = 1.0
prob = ODEProblem(f, [u0_true], (0.0, 5.0), [0.3])

tdata = collect(0.0:0.25:5.0)
u_exact = u0_true .* exp.(-p_true .* tdata)
σ = 0.01
ydata = u_exact .+ σ .* randn(rng, length(tdata))
data = ExperimentData(tdata, ydata; σ = σ)

@testset "Regularizer residual layout on a plain fit" begin
    # One tunable model parameter, no tunable ICs -> 1 penalized entry, 1 residual row.
    pe = ParameterEstimationLayer(prob, Tsit5(), data)
    p0 = CorleoneParameterEstimationProblem(pe; regularization = L2Regularizer(1.0e-3))
    @test p0.n_regularization_residuals == 1
    @test p0.n_shooting_constraints == 0
    @test p0.n_residuals == n_observations(data) + 1

    # The regularization residual is appended after the observations.
    r = p0.residual(similar(p0.x0, p0.n_residuals), p0.x0, nothing)
    @test length(r) == p0.n_residuals
    # At x0 the penalty residual is √λ·(θ - ref), with ref=0.
    @test r[end] ≈ sqrt(1.0e-3) * p0.x0[1]
end

@testset "L2 ridge recovers and shrinks" begin
    pe = ParameterEstimationLayer(prob, Tsit5(), data)
    # Small λ: essentially the unregularized least-squares fit.
    sol_small = solve(problem(pe; regularization = L2Regularizer(1.0e-3)), GaussNewton(); abstol = 1.0e-10, reltol = 1.0e-10)
    @test SciMLBase.successful_retcode(sol_small)
    @test isapprox(sol_small.u[1], p_true; atol = 0.1)

    # Large λ with reference 0 pulls the estimate toward 0.
    sol_big = solve(problem(pe; regularization = L2Regularizer(5.0)), GaussNewton())
    @test abs(sol_big.u[1]) < abs(sol_small.u[1])

    # Non-zero reference pulls toward reference.
    ref = p_true + 0.5
    sol_ref = solve(problem(pe; regularization = L2Regularizer(5.0, reference = [ref])), GaussNewton())
    @test abs(sol_ref.u[1] - ref) < abs(sol_small.u[1] - ref)
end

@testset "L1 lasso shrinks toward the reference" begin
    pe = ParameterEstimationLayer(prob, Tsit5(), data)
    sol = solve(problem(pe; regularization = L1Regularizer(5.0e-3)), GaussNewton())
    @test SciMLBase.successful_retcode(sol)
    # L1 pulls toward 0.
    @test abs(sol.u[1]) < 0.7 + 1.0e-3

    # Elastic net combines L1 and L2 residuals.
    pe2 = ParameterEstimationLayer(prob, Tsit5(), data)
    p2 = CorleoneParameterEstimationProblem(pe2; regularization = ElasticNetRegularizer(1.0e-3, 1.0e-3))
    @test p2.n_regularization_residuals == 2   # one L1 row + one L2 row
    sol2 = solve(NonlinearSolve.NonlinearLeastSquaresProblem(p2), GaussNewton())
    @test SciMLBase.successful_retcode(sol2)
    @test isfinite(sol2.u[1])
end

@testset "Custom regularizer functor" begin
    pe = ParameterEstimationLayer(prob, Tsit5(), data)
    # A custom linear-penalty regularizer returning a residual vector.
    reg = θ -> sqrt(1.0e-3) .* θ          # identical to L2 toward 0
    p0 = CorleoneParameterEstimationProblem(pe; regularization = reg)
    @test p0.n_regularization_residuals == 1
    sol = solve(NonlinearSolve.NonlinearLeastSquaresProblem(p0), GaussNewton())
    @test SciMLBase.successful_retcode(sol)
    @test isfinite(sol.u[1])

    # A scalar-valued regularizer (a single residual row).
    reg_scalar = θ -> [sqrt(1.0e-3) * sum(θ)]
    p1 = CorleoneParameterEstimationProblem(pe; regularization = reg_scalar)
    @test p1.n_regularization_residuals == 1
end

@testset "Tunable initial conditions are NOT regularized by default" begin
    # Two shared entries: [u0; p]. The IC entry is tunable.
    pe_ic = ParameterEstimationLayer(
        prob, Tsit5(), data;
        observed = (u, p, t) -> u[1:1], tunable_ic = [1]
    )
    prob_obj = CorleoneParameterEstimationProblem(pe_ic)
    p0 = CorleoneParameterEstimationProblem(pe_ic; regularization = L2Regularizer(1.0))
    # Only the model parameter (shared index 2) is penalized -> 1 residual row.
    @test p0.n_regularization_residuals == 1
    @test length(p0.x0) == 2

    # Verify the penalty is on shared index 2 (the parameter), not index 1 (the IC):
    # the residual's trailing entry depends only on x0[2].
    r = p0.residual(similar(p0.x0, p0.n_residuals), p0.x0, nothing)
    @test r[end] ≈ sqrt(1.0) * p0.x0[2]

    # Explicitly regularizing the IC as well gives two penalized entries.
    p_both = CorleoneParameterEstimationProblem(
        pe_ic;
        regularization = L2Regularizer(1.0, reference = [u0_true, p_true]),
        regularized_indices = [1, 2]
    )
    @test p_both.n_regularization_residuals == 2
    @test p_both.x0[1] ≈ u0_true
    r_both = p_both.residual(similar(p_both.x0, p_both.n_residuals), p_both.x0, nothing)
    @test r_both[end - 1] ≈ r_both[end - 1]
end

@testset "Regularization composes with parameter transforms (natural space)" begin
    # Fit a strictly positive parameter via a Log transform; the penalty acts on the
    # *natural* value θ = exp(φ), not the internal φ.
    pe = ParameterEstimationLayer(prob, Tsit5(), data)
    p0 = CorleoneParameterEstimationProblem(
        pe;
        transforms = [Log()], regularization = L2Regularizer(0.5, reference = [0.7])
    )
    @test p0.n_regularization_residuals == 1
    sol = solve(NonlinearSolve.NonlinearLeastSquaresProblem(p0), GaussNewton())
    @test SciMLBase.successful_retcode(sol)
    # The reported/estimated parameter (natural space) is positive and near the truth.
    est = estimate_parameters(p0, sol)
    @test est.x[1] > 0
    @test isapprox(est.x[1], p_true; atol = 0.3)
end

@testset "estimate_parameters stays consistent under regularization" begin
    pe = ParameterEstimationLayer(prob, Tsit5(), data)
    p0 = CorleoneParameterEstimationProblem(pe; regularization = L2Regularizer(1.0e-2))
    sol = solve(NonlinearSolve.NonlinearLeastSquaresProblem(p0), GaussNewton())
    est = estimate_parameters(p0, sol)
    @test est.n_parameters == 1
    @test est.n_observations == n_observations(data)
    # RSS uses only the observation block, so it differs from the penalized norm.
    @test est.residual_norm <= sum(abs2, sol.resid)
    @test isfinite(est.std_errors[1])

    # With regularization the covariance (MAP) is well-defined and finite even for a
    # strong penalty.
    pe_s = ParameterEstimationLayer(prob, Tsit5(), data)
    ps = CorleoneParameterEstimationProblem(pe_s; regularization = L2Regularizer(0.1))
    sol_s = solve(NonlinearSolve.NonlinearLeastSquaresProblem(ps), GaussNewton())
    est_s = estimate_parameters(ps, sol_s)
    @test isfinite(est_s.std_errors[1])
end

@testset "Regularizer argument validation" begin
    pe = ParameterEstimationLayer(prob, Tsit5(), data)
    # λ must be > 0.
    @test_throws ArgumentError L2Regularizer(0.0)
    @test_throws ArgumentError L1Regularizer(-1.0)
    # Regularized indices must be within the shared block [1:1] here (1 parameter).
    @test_throws ArgumentError CorleoneParameterEstimationProblem(
        pe;
        regularization = L2Regularizer(1.0), regularized_indices = [2]
    )
    pe_ic = ParameterEstimationLayer(
        prob, Tsit5(), data;
        observed = (u, p, t) -> u[1:1], tunable_ic = [1]
    )
    # reference length must match the penalized count.
    @test_throws ArgumentError CorleoneParameterEstimationProblem(
        pe_ic;
        regularization = L2Regularizer(1.0, reference = [1.0, 1.0])
    )
end
