using Corleone, LuxCore, OrdinaryDiffEqTsit5, Random, Test, ComponentArrays
using SciMLBase, SciMLSensitivity, ForwardDiff, ReverseDiff, Zygote
using DiffEqBase: SensitivityADPassThrough

include("../shared/shooting_ad_case.jl")
using .ShootingADCase

@testset "Full-solver gradients against an analytic reference" begin
    for (name, gradient, pass_through, algorithms) in (
            (:ForwardDiff, ForwardDiff.gradient, false, (EnsembleSerial(), EnsembleThreads())),
            (:ReverseDiff, ReverseDiff.gradient, true, (EnsembleSerial(),)),
            (:Zygote, (f, q) -> only(Zygote.gradient(f, q)), false, (EnsembleSerial(), EnsembleThreads())),
        )
        @testset "$name / $(typeof(ensemblealg))" for ensemblealg in algorithms
            loss = backend_case(pass_through, ensemblealg)
            for q in ([0.2, 1.0, 1.5, 2.5, 3.0], [0.3, 1.2, 2.0, 3.5, 4.0])
                original = copy(q)
                @test loss(q) ≈ analytic_loss(q) rtol = 1.0e-8
                @test gradient(loss, q) ≈ ForwardDiff.gradient(analytic_loss, q) rtol = 1.0e-7 atol = 1.0e-8
                @test q == original
            end
        end
    end
end
