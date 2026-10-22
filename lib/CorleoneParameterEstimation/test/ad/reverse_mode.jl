using Corleone, CorleoneParameterEstimation
using OrdinaryDiffEqTsit5
using LuxCore
using ForwardDiff
using Zygote
using ReverseDiff
using SciMLSensitivity
using SciMLBase
using Optimization, OptimizationLBFGSB
using NonlinearSolve
using ComponentArrays
using Test
using Random
using StableRNGs

# Reverse-mode automatic differentiation through a parameter-estimation fit. The in-place
# `residuals!` path cannot be differentiated by Zygote (it mutates a buffer and the
# shooting constraints), so the package also provides an out-of-place residual. These
# tests exercise that path directly and through the Optimization.jl extension with
# `AutoZygote`, and check the reverse-mode gradient against forward-mode `ForwardDiff`.
#
# `SciMLSensitivity` provides the adjoint rule for differentiating the ODE solve; it is a
# dependency of this test group only (not of the package).

rng = StableRNG(42)

# Model: du/dt = -p[1]*u, exact solution u(t) = u0 * exp(-p[1]*t).
f = (du, u, p, t) -> (du[1] = -p[1] * u[1]; nothing)
p_true = 0.7
prob = ODEProblem(f, [1.0], (0.0, 5.0), [0.3])
tdata = collect(0.0:0.25:5.0)
ydata = exp.(-p_true .* tdata)
data = ExperimentData(tdata, ydata; σ = 0.01)
pe = ParameterEstimationLayer(prob, Tsit5(), data)
pe_problem = CorleoneParameterEstimation.CorleoneParameterEstimationProblem(pe)

@testset "out-of-place residual is reverse-mode differentiable" begin
    loss(x) = 0.5 * sum(abs2, pe_problem.residual_oop(x, nothing))
    g_zygote = Zygote.gradient(loss, pe_problem.x0)[1]
    g_forward = ForwardDiff.gradient(loss, pe_problem.x0)
    @test g_zygote ≈ g_forward rtol = 1e-6
end

@testset "Optimization extension with AutoZygote" begin
    optprob = OptimizationProblem(pe; AD = AutoZygote())

    # Reverse-mode gradient of the scalar objective matches forward mode.
    g_zygote = Zygote.gradient(x -> optprob.f(x, nothing), optprob.u0)[1]
    g_forward = ForwardDiff.gradient(x -> optprob.f(x, nothing), optprob.u0)
    @test g_zygote ≈ g_forward rtol = 1e-6

    # A reverse-mode-driven solver recovers the true parameter.
    sol = solve(optprob, LBFGSB())
    @test SciMLBase.successful_retcode(sol)
    @test isapprox(sol.u[1], p_true; atol = 0.05)
end

@testset "Optimization extension with AutoReverseDiff" begin
    # ReverseDiff traces the ODE solve itself, so the problem must disable
    # SciMLSensitivity's adjoint (`sensealg = NoAD()`). With an adjoint sensealg the solve
    # returns a `ReverseDiff.TrackedArray` instead of an `ODESolution`, and the Corleone
    # layer needs the `ODESolution` interface (`.u`, `.t`).
    prob_rd = ODEProblem(f, [1.0], (0.0, 5.0), [0.3]; sensealg = SciMLBase.NoAD())
    pe_rd = ParameterEstimationLayer(prob_rd, Tsit5(), data)
    optprob = OptimizationProblem(pe_rd; AD = AutoReverseDiff())

    g_reverse = ReverseDiff.gradient(x -> optprob.f(x, nothing), optprob.u0)
    g_forward = ForwardDiff.gradient(x -> optprob.f(x, nothing), optprob.u0)
    @test g_reverse ≈ g_forward rtol = 1e-6

    sol = solve(optprob, LBFGSB())
    @test SciMLBase.successful_retcode(sol)
    @test isapprox(sol.u[1], p_true; atol = 0.05)
end

@testset "Optimization extension with AutoReverseDiff and multiple shooting" begin
    # Multiple shooting under reverse mode. The out-of-place residual is used by the
    # extension; unlike Zygote, ReverseDiff's operator-overloading tape tolerates the
    # mutation in the root package's multiple-shooting trajectory assembly, so no extra
    # `_concretize_residual` type assertion is required (it is skipped for non-
    # `AbstractFloat` element types such as `ReverseDiff.TrackedReal`).
    prob_rd = ODEProblem(f, [1.0], (0.0, 5.0), [0.3]; sensealg = SciMLBase.NoAD())
    shooting_points = collect(0.0:1.0:5.0)
    pe_ms = ParameterEstimationLayer(
        prob_rd, Tsit5(), data, shooting_points; observed = (u, p, t) -> u[1:1]
    )
    pe_problem_ms = CorleoneParameterEstimation.CorleoneParameterEstimationProblem(pe_ms)

    loss_ms(x) = 0.5 * sum(abs2, pe_problem_ms.residual_oop(x, nothing))
    g_reverse = ReverseDiff.gradient(loss_ms, pe_problem_ms.x0)
    g_forward = ForwardDiff.gradient(loss_ms, pe_problem_ms.x0)
    @test g_reverse ≈ g_forward rtol = 1e-6

    optprob = OptimizationProblem(pe_ms; AD = AutoReverseDiff())
    sol = solve(optprob, LBFGSB())
    @test SciMLBase.successful_retcode(sol)
    # The first optimization variable is the shared model parameter.
    @test isapprox(sol.u[1], p_true; atol = 0.05)
end

@testset "Parameter transforms under reverse mode" begin
    # A `Log` transform makes the internal problem unconstrained, so `problem(pe; transforms)`
    # drops the bounds and reverse-mode AD through NonlinearSolve keeps working. The
    # out-of-place residual (now `exp`-based) is still reverse-mode differentiable.
    #
    # ReverseDiff traces the solve itself, so it needs the `NoAD` sensealg (as above).
    prob_rd = ODEProblem(f, [1.0], (0.0, 5.0), [0.3]; sensealg = SciMLBase.NoAD())
    pe_rd = ParameterEstimationLayer(prob_rd, Tsit5(), data)
    pe_rd_log_problem = CorleoneParameterEstimation.CorleoneParameterEstimationProblem(
        pe_rd; transforms = [Log()]
    )
    loss_log(x) = 0.5 * sum(abs2, pe_rd_log_problem.residual_oop(x, nothing))
    g_reverse = ReverseDiff.gradient(loss_log, pe_rd_log_problem.x0)
    g_forward = ForwardDiff.gradient(loss_log, pe_rd_log_problem.x0)
    # The internal `Log` reparameterization scales the gradient by `θ` (and the residuals by
    # `1/σ`), so a slightly looser relative tolerance is used than for the O(1) gradients above.
    @test g_reverse ≈ g_forward rtol = 1e-5

    nlls_log = problem(pe; transforms = [Log()])
    @test isnothing(nlls_log.lb) && isnothing(nlls_log.ub)
    sol_log = solve(
        nlls_log, GaussNewton(; autodiff = AutoZygote()); abstol = 1e-8, reltol = 1e-8
    )
    @test SciMLBase.successful_retcode(sol_log)
    # The optimizer works in log space; map back to the natural parameter.
    @test isapprox(exp(sol_log.u[1]), p_true; atol = 0.05)
end

@testset "NonlinearSolve with AutoZygote" begin
    # `problem(pe)` drops trivial (`±Inf`) bounds, so the unbounded solvers can be used
    # with reverse-mode AD. Passing any `lb`/`ub` (even trivial ones) makes NonlinearSolve
    # wrap the residual in a mutating bounds transform that Zygote cannot differentiate.
    nlls = problem(pe)
    @test isnothing(nlls.lb) && isnothing(nlls.ub)

    sol = solve(nlls, GaussNewton(; autodiff = AutoZygote()); abstol = 1e-8, reltol = 1e-8)
    @test SciMLBase.successful_retcode(sol)
    @test isapprox(sol.u[1], p_true; atol = 0.05)

    # Finite bounds are kept; solve with a `Bounded*` solver, which handles bounds natively
    # (no mutating transform) and therefore stays reverse-mode differentiable.
    layer_b = SingleShootingLayer(
        prob, Tsit5(); saveat = unique(tdata), bounds_p = ([0.0], [2.0])
    )
    pe_b = ParameterEstimationLayer(layer_b, data)
    nlls_b = problem(pe_b)
    @test nlls_b.lb == [0.0] && nlls_b.ub == [2.0]

    sol_b = solve(
        nlls_b, BoundedGaussNewton(; autodiff = AutoZygote()); abstol = 1e-8, reltol = 1e-8
    )
    @test isapprox(sol_b.u[1], p_true; atol = 0.05)
end
