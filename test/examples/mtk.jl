using Test
using Corleone
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using OrdinaryDiffEqTsit5
using ComponentArrays, ForwardDiff
using Optimization
using OptimizationMOI, Ipopt
using LuxCore, Random


@variables x(..) = 0.5 [tunable = false] y(..) = 0.7 [tunable = false]
@variables u(..) = 0.0 [bounds = (0.0, 1.0), input = true]
@constants begin
    c₁ = 0.4
    c₂ = 0.2
end
@parameters begin
    α = 1.0, [tunable = false]
    β = 1.0, [tunable = true, bounds = (0.9, 1.1)]
end

cost = [
    Symbolics.Integral(t in (0.0, 12.0))(
        (x(t) - 1.0)^2 + (y(t) - 1.0)^2
    ),
]

cons = [
    x(0.0) ≳ 0.2,
    β ~ 1.0,
]

@named lotka = System(
    [
        D(x(t)) ~ x(t) - β * x(t) * y(t) - c₁ * u(t) * x(t),
        D(y(t)) ~ - y(t) + x(t) * y(t) - c₂ * u(t) * y(t),
    ], t; costs = cost, constraints = cons
)

@testset "Single Shooting" begin
    dynopt = CorleoneDynamicOptProblem(
        lotka, [],
        u(t) => 0.0:0.1:11.9,
        algorithm = Tsit5(),
    )

    optprob = OptimizationProblem(dynopt, AutoForwardDiff(), Val(:ComponentArrays))

    @test size(optprob.lcons, 1) == size(optprob.ucons, 1) == length(cons)

    ps, st = LuxCore.setup(Random.default_rng(), dynopt.layer)

    traj, _ = dynopt.layer(nothing, ps, st)

    vars = map(dynopt.getters) do get
        get(traj)
    end

    @test dynopt.objective(ps, st) ≈ optprob.f(optprob.u0, optprob.p)
    @test isapprox(dynopt.objective(ps, st), 6.062277381976436, atol = 1.0e-4)

    sol = solve(
        optprob, Ipopt.Optimizer(), max_iter = 1000, tol = 5.0e-6,
        hessian_approximation = "limited-memory"
    )

    @test isapprox(sol.u[1], 1.0, atol = 1.0e-4)
    @test SciMLBase.successful_retcode(sol)
    @test isapprox(sol.objective, 1.344336, atol = 1.0e-4)
end

@testset "Multiple Shooting" begin
    dynopt = CorleoneDynamicOptProblem(
        lotka, [],
        u(t) => 0.0:0.1:11.9,
        algorithm = Tsit5(),
        shooting = [0.0, 3.0, 6.0, 9.0]
    )

    optprob = OptimizationProblem(dynopt, AutoForwardDiff(), Val(:ComponentArrays))

    @test size(optprob.lcons, 1) == size(optprob.ucons, 1) == length(cons) + Corleone.get_number_of_shooting_constraints(dynopt.layer)

    ps, st = LuxCore.setup(Random.default_rng(), dynopt.layer)

    traj, _ = dynopt.layer(nothing, ps, st)

    vars = map(dynopt.getters) do get
        get(traj)
    end

    @test dynopt.objective(ps, st) ≈ optprob.f(optprob.u0, optprob.p)
    @test isapprox(dynopt.objective(ps, st), 1.2417260078523538, atol = 1.0e-4)

    sol = solve(
        optprob, Ipopt.Optimizer(), max_iter = 1000, tol = 5.0e-6,
        hessian_approximation = "limited-memory"
    )

    @test isapprox(sol.u[1], 1.0, atol = 1.0e-4)
    @test SciMLBase.successful_retcode(sol)
    @test isapprox(sol.objective, 1.344336, atol = 1.0e-4)
end
