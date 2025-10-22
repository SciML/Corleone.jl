using Corleone
using OrdinaryDiffEqTsit5
using Test
using Random
using LuxCore
using ComponentArrays
using LinearAlgebra

rng = Random.default_rng()

function lotka_dynamics(u, p, t)
    return [u[1] - p[2] * prod(u[1:2]) - 0.4 * p[1] * u[1];
            -u[2] + p[3] * prod(u[1:2]) - 0.2 * p[1] * u[2]]
end

tspan = (0., 12.)
u0 = [0.5, 0.7]
p0 = [0.0, 1.0, 1.0]

prob = ODEProblem(lotka_dynamics, u0, tspan, p0; abstol=1e-8, reltol=1e-6)

@testset "Continuous case" begin
    layer_all_p = OEDLayer(prob, Tsit5(); params=[2,3], observed = (u,p,t) -> u[1:2])
    layer_p1 = OEDLayer(prob, Tsit5(); params=[2], observed = (u,p,t) -> u[1:2])
    layer_p2 = OEDLayer(prob, Tsit5(); params=[3], observed = (u,p,t) -> u[1:2])

    multilayer_all_p = MultiExperimentLayer(layer_all_p, 1)
    multilayer_p1_p2 = MultiExperimentLayer(layer_p1, layer_p2)

    F_all_1 = Corleone.observed_sensitivity_product_variables(multilayer_all_p.layers.layer, 1)
    F_all_2 = Corleone.observed_sensitivity_product_variables(multilayer_all_p.layers.layer, 2)

    F_p1_1 = Corleone.observed_sensitivity_product_variables(multilayer_p1_p2.layers[1].layer, 1)
    F_p1_2 = Corleone.observed_sensitivity_product_variables(multilayer_p1_p2.layers[1].layer, 2)

    F_p2_1 = Corleone.observed_sensitivity_product_variables(multilayer_p1_p2.layers[2].layer, 1)
    F_p2_2 = Corleone.observed_sensitivity_product_variables(multilayer_p1_p2.layers[2].layer, 2)


    rng = Random.default_rng()
    ps_all_p, st_all_p = LuxCore.setup(rng, multilayer_all_p)
    ps_1_2, st_1_2 = LuxCore.setup(rng, multilayer_p1_p2)

    crit = ACriterion()

    ACrit_all_p = crit(multilayer_all_p)
    ACrit_p1_p2 = crit(multilayer_p1_p2)

    phi_all = ACrit_all_p(ComponentArray(ps_all_p), nothing)
    phi_p1_p2 = ACrit_p1_p2(ComponentArray(ps_1_2), nothing)

    # Assemble the FIMs manually
    sols_all_p, _ = multilayer_all_p(nothing, ps_all_p, st_all_p)
    sols_p1_p2, _ = multilayer_p1_p2(nothing, ps_1_2, st_1_2)

    F11_all = Corleone.symmetric_from_vector(sols_all_p[1][F_all_1][end])
    F22_all = Corleone.symmetric_from_vector(sols_all_p[1][F_all_2][end])

    F11_p1_h1 = Corleone.symmetric_from_vector(sols_p1_p2[1][F_p1_1][end])
    F11_p1_h2 = Corleone.symmetric_from_vector(sols_p1_p2[1][F_p1_2][end])
    F22_p2_h1 = Corleone.symmetric_from_vector(sols_p1_p2[2][F_p2_1][end])
    F22_p2_h2 = Corleone.symmetric_from_vector(sols_p1_p2[2][F_p2_2][end])

    @test isapprox(F11_all[1,1], only(F11_p1_h1))
    @test isapprox(F22_all[1,1], only(F11_p1_h2))
    @test isapprox(F11_all[2,2], only(F22_p2_h1))
    @test isapprox(F22_all[2,2], only(F22_p2_h2))


    @test isapprox(crit(F11_all + F22_all), phi_all , atol=1e-8)
    @test isapprox(crit(F11_p1_h1+F11_p1_h2+F22_p2_h1+F22_p2_h2), phi_p1_p2, atol=1e-8)
end

@testset "Discrete case" begin
    layer_all_p = OEDLayer(prob, Tsit5(); params=[2,3], observed = (u,p,t) -> u[1:2], measurement_points = 0.5:1.0:11.5)
    layer_p1 = OEDLayer(prob, Tsit5(); params=[2], observed = (u,p,t) -> u[1:2], measurement_points = 0.5:1.0:11.5)
    layer_p2 = OEDLayer(prob, Tsit5(); params=[3], observed = (u,p,t) -> u[1:2], measurement_points = 0.5:1.0:11.5)

    multilayer_all_p = MultiExperimentLayer(layer_all_p, 1)
    multilayer_p1_p2 = MultiExperimentLayer(layer_p1, layer_p2)

    ps, st = LuxCore.setup(Random.default_rng(), multilayer_all_p)
    ps1, st1 = LuxCore.setup(Random.default_rng(), multilayer_p1_p2)

    F_all_p = Corleone.fim(multilayer_all_p, ComponentArray(ps))

    F_p1_p2 = Corleone.fim(multilayer_p1_p2, ComponentArray(ps1))

    @test isapprox(tr(F_all_p),tr(F_p1_p2), atol=1e-5)

end