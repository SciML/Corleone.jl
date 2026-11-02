using Corleone, OrdinaryDiffEqTsit5, LuxCore, ComponentArrays, Random, Test
using Optimization, OptimizationMOI, Ipopt, ForwardDiff
using SymbolicIndexingInterface

@testset "Lotka single shooting with explicit optimization" begin
    function dynamics!(du, u, p, t)
        du[1] = u[1] - p[2] * u[1] * u[2] - 0.4 * p[1] * u[1]
        du[2] = -u[2] + p[3] * u[1] * u[2] - 0.2 * p[1] * u[2]
        du[3] = (u[1] - 1)^2 + (u[2] - 1)^2
        return nothing
    end
    prob = ODEProblem(
        dynamics!, [0.5, 0.7, 0.0], (0.0, 12.0), [0.0, 1.0, 1.0];
        abstol = 1.0e-9, reltol = 1.0e-9
    )
    layer = ShootingLayer(
        prob, Tsit5(),
        Tunable(1; kind = Val(:parameter)) => collect(0.1:0.1:11.9),
        Constant(2; kind = Val(:parameter)) => nothing,
        Constant(3; kind = Val(:parameter)) => nothing
    )
    ps, st = LuxCore.setup(MersenneTwister(42), layer)
    trajectory, _ = layer(nothing, ps, st)
    @test trajectory[:u₁] == first.(trajectory.u)
    @test trajectory[:p₁] == zeros(120)
    @test allunique(trajectory.t)
    @test all(isempty, trajectory.residuals)
    @test LuxCore.parameterlength(layer) == 120
    @test last(trajectory.u)[3] ≈ 6.062277454291031 atol = 1.0e-5

    initial = ComponentArray(ps)
    axes = getaxes(initial)
    objective(q, state) = last(first(layer(nothing, ComponentArray(q, axes), state)).u)[3]
    optfun = OptimizationFunction(objective, AutoForwardDiff())
    optprob = OptimizationProblem(
        optfun, initial, st;
        lb = zeros(length(initial)), ub = ones(length(initial))
    )
    @test optprob.f(optprob.u0, optprob.p) ≈ last(trajectory.u)[3]
    result = solve(
        optprob, Ipopt.Optimizer(); max_iter = 300, tol = 1.0e-6,
        hessian_approximation = "limited-memory", print_level = 0
    )
    @test SciMLBase.successful_retcode(result)
    @test result.objective ≈ 1.344336 atol = 1.0e-4
    optimized, _ = layer(nothing, ComponentArray(result.u, axes), st)
    @test last(optimized.u)[3] ≈ result.objective
    @test all(x -> -1.0e-7 <= x <= 1 + 1.0e-7, result.u)
end
