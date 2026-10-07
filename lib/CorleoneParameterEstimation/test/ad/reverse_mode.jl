using Corleone, CorleoneParameterEstimation
using OrdinaryDiffEqTsit5
using LuxCore
using ForwardDiff
using Zygote
using ReverseDiff
using SciMLSensitivity
using SciMLBase
using Optimization, OptimizationLBFGSB
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
problem = CorleoneParameterEstimation.CorleoneParameterEstimationProblem(pe)

@testset "out-of-place residual is reverse-mode differentiable" begin
    loss(x) = 0.5 * sum(abs2, problem.residual_oop(x, nothing))
    g_zygote = Zygote.gradient(loss, problem.x0)[1]
    g_forward = ForwardDiff.gradient(loss, problem.x0)
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
