using CorleoneOED
using OrdinaryDiffEqTsit5
using Test
using StableRNGs
using LuxCore
using ComponentArrays
using Optimization, OptimizationMOI, Ipopt

rng = StableRNG(1111)

function lin_dyn(u, p, t)
    return [p[1] * u[1]]
end

u0 = [1.0]
tspan = (0.0, 1.0)
p = [-2.0]

prob = ODEProblem(lin_dyn, u0, tspan, p)
ol = ShootingLayer(prob, Tsit5(); fixed_parameters = [1])
oed = OEDLayer(
    ol; params = [1], measurements = (
        ContinuousMeasurement(:weight, collect(0.0:0.01:0.99), (u, p, t) -> u[1]),
    )
)
ps, st = LuxCore.setup(rng, oed)
lb, ub = Corleone.get_bounds(oed, ps, st) .|> ComponentArray
@test_nowarn @inferred oed(nothing, ps, st)
traj_oed, _ = oed(nothing, ps, st)
@test traj_oed.t == 0.0:0.01:1.0
@test fisher_information(oed, traj_oed) ≈ first(fisher_information(oed, nothing, ps, st))
@test reduce(vcat, first(observed_equations(oed, nothing, ps, st))) == first.(traj_oed.u)
@test LuxCore.parameterlength(oed) == LuxCore.parameterlength(ol) + 100
@test all(iszero, lb)
@test all(isone, ub)

@testset "Criteria" begin
    foreach(
        (
            ACriterion(), DCriterion(), ECriterion(),
            FisherACriterion(), FisherDCriterion(), FisherECriterion(),
        )
    ) do crit
        @test_nowarn @inferred crit(oed, nothing, ps, st)
    end
end

@testset "Information Gain" begin
    optprob = OptimizationProblem(oed, ACriterion(); M = [0.2])
    uopt = solve(
        optprob, Ipopt.Optimizer(),
        tol = 1.0e-10,
        hessian_approximation = "limited-memory",
        max_iter = 300,
        print_level = 3,
    )

    popt = uopt + zero(ComponentArray(ps))

    μ = uopt.original.inner.mult_g
    traj, _ = oed(nothing, popt, st)
    Π, _ = CorleoneOED.global_information_gain(oed, nothing, popt, st)
    optimality = reduce(vcat, map(xi -> only(first(xi)), Π))
    idx = findall(optimality .> μ)
    @test !isempty(idx)
    @test all(0.4 .< traj.t[idx] .<= 0.6)
    @test all(popt.parameter.parameters.weight.parameter.ps[idx] .> 0.99)
end

@testset "Optimization" begin
    for MEASUREMENT in (true, false)
        measurement = MEASUREMENT ? DiscreteMeasurement : ContinuousMeasurement
        oed = OEDLayer(
            ol; params = [1], measurements = (
                measurement(:weight, collect(0.0:0.1:0.9), (u, p, t) -> [u[1]]),
            )
        )
        ps, st = LuxCore.setup(rng, oed)
        uref = MEASUREMENT ? vcat(zeros(5), [1.0, 1.0], zeros(3)) : vcat(zeros(4), [1.0, 1.0], zeros(4))

        ucon = MEASUREMENT ? 2.0 : 0.2
        for crit in (ACriterion(), DCriterion(), ECriterion())
            optprob = OptimizationProblem(oed, crit; M = [ucon])
            uopt = solve(
                optprob, Ipopt.Optimizer(),
                tol = 1.0e-10,
                hessian_approximation = "limited-memory",
                max_iter = 300,
                print_level = 0,
            )

            popt = uopt + zero(ComponentArray(ps))
            @test SciMLBase.successful_retcode(uopt)
            @test isapprox(collect(popt), uref, atol = 1.0e-3)
            @test isapprox(CorleoneOED.get_sampling_sums(oed, nothing, popt, st), [ucon], atol = 1.0e-3)
        end
    end
end
