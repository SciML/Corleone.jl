using Corleone
using OrdinaryDiffEqTsit5
using Test
using Random
using LuxCore
using Symbolics
using ForwardDiff
using LinearAlgebra
using ComponentArrays

crits = [ACriterion(), DCriterion(), ECriterion(),
            FisherACriterion(), FisherDCriterion(), FisherECriterion(),]

@testset "Simple evaluation of criteria" begin
    F = diagm(ones(10))
    @test [crit(F) for crit in crits] == [10.0, 1.0, 1.0, -10.0, -1.0, -1.0]
    @test [crit(Symmetric(F)) for crit in crits] == [10.0, 1.0, 1.0, -10.0, -1.0, -1.0]
end


@testset "Single shooting criteria fixed and with controls" begin
    # Linear system with control p[2] and Mayer objective
    function lin1d(u,p,t)
        return [- p[1] * u[1] + p[2];
                (u[1]-1.0)^2]
    end

    u0 = [0.5, 0.0]
    p = [1.0, 0.0]
    tspan = (0.,1.)

    prob = ODEProblem(lin1d, u0, tspan, p)
    sol = solve(prob, Tsit5())

    control = ControlParameter(0.0:0.01:0.99, name=:control, bounds=(-1.,1.), controls=zeros(100))
    observed = (u,p,t) -> [u[1]]
    oedlayer_wo_c = OEDLayer(prob, Tsit5(); params=[1], dt=0.1, observed=observed)
    oedlayer_w_c = OEDLayer(prob, Tsit5(); params=[1], dt=0.1, observed=observed,
                        control_indices = [2], controls=(control,))
    rng = Random.default_rng()
    ps_wo_c, st_w_c = LuxCore.setup(rng, oedlayer_wo_c)
    ps_w_c, st_wo_c = LuxCore.setup(rng, oedlayer_w_c)
    p_wo_c = ComponentArray(ps_wo_c)
    p_w_c = ComponentArray(ps_w_c)

    F_init = 49.48602824180152

    for crit in crits
        if crit in [ACriterion(), DCriterion(), ECriterion()]
            @test isapprox(crit(oedlayer_w_c)(p_w_c, nothing), F_init, atol=1e-6)
            @test isapprox(crit(oedlayer_wo_c)(p_wo_c, nothing), F_init, atol=1e-6)
        else
            @test isapprox(crit(oedlayer_w_c)(p_w_c, nothing), -inv(F_init), atol=1e-6)
            @test isapprox(crit(oedlayer_wo_c)(p_wo_c, nothing), -inv(F_init), atol=1e-6)
        end
    end

end