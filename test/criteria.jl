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

@testset "FIM: Multiple dispatch" begin
    function lotka(u, p, t)
        return [u[1] - p[2] * prod(u[1:2]) - 0.4 * p[1] * u[1];
                -u[2] + p[3] * prod(u[1:2]) - 0.2 * p[1] * u[2]]
    end

    tspan = (0., 12.)
    u0 = [0.5, 0.7, ]
    p0 = [0.0, 1.0, 1.0]
    prob = ODEProblem{false}(lotka, u0, tspan, p0,
        sensealg = ForwardDiffSensitivity()
        )
    control = ControlParameter(
        collect(0.0:0.25:11.75), name = :fishing, bounds = (0.,1.)
    )

    dt = 0.1
    oed1 = Corleone.OEDLayer(prob, Tsit5(); params=[2,3], controls = (control,),
                control_indices = [1], dt = dt)
    p1, s1 = LuxCore.setup(Random.default_rng(), oed1)

    shooting_points = [0.0,4.0, 8.0, 12.0]
    oed2 = Corleone.OEDLayer(prob, Tsit5(), shooting_points; params=[2,3], controls = (control,),
                control_indices = [1], dt = dt)

    p2, s2 = LuxCore.setup(Random.default_rng(), oed2)
    p2_fwd, _ = ForwardSolveInitialization()(Random.default_rng(), oed2)

    measurement_points = 12*rand(10) |> sort
    oed3 = Corleone.OEDLayer(prob, Tsit5(); params=[2,3], controls = (control,),
                control_indices = [1], measurement_points = measurement_points)
    p3, s3 = LuxCore.setup(Random.default_rng(), oed3)

    oed4 = Corleone.OEDLayer(prob, Tsit5(), shooting_points; params=[2,3], controls = (control,),
                control_indices = [1], measurement_points=measurement_points)

    p4, s4 = LuxCore.setup(Random.default_rng(), oed4)
    p4_fwd, _ = ForwardSolveInitialization()(Random.default_rng(), oed4)

    fim1 = Corleone.fim(oed1)
    fim2 = Corleone.fim(oed2)
    fim2 = Corleone.fim(oed2)
    fim3 = Corleone.fim(oed3)
    fim4 = Corleone.fim(oed4)
    F1 = fim1(ComponentArray(p1),nothing)
    F2 = fim2(ComponentArray(p2),nothing)
    F2_fwd = fim2(ComponentArray(p2_fwd),nothing)
    F3 = fim3(ComponentArray(p3),nothing)
    F4 = fim4(ComponentArray(p4),nothing)
    F4_fwd = fim4(ComponentArray(p4_fwd),nothing)

    @test isapprox(F1, F2_fwd)
    @test isapprox(F3, F4_fwd)
end
@testset "Continuous OED: fixed and with controls" begin
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
    ps_wo_c, st_wo_c = LuxCore.setup(rng, oedlayer_wo_c)
    ps_w_c, st_w_c = LuxCore.setup(rng, oedlayer_w_c)
    p_wo_c = ComponentArray(ps_wo_c)
    p_w_c = ComponentArray(ps_w_c)

    C_init = 49.48602824180152

    @test isapprox(Corleone.fim(oedlayer_w_c, p_w_c)[1,1], inv(C_init), atol=1e-4)
    @test isapprox(Corleone.fim(oedlayer_wo_c, p_wo_c)[1,1], inv(C_init), atol=1e-4)

    for crit in crits
        sols_w_c, _ = oedlayer_w_c(nothing, p_w_c, st_w_c)

        if crit in [ACriterion(), DCriterion(), ECriterion()]
            @test isapprox(crit(oedlayer_w_c)(p_w_c, nothing), C_init, atol=1e-6)
            @test isapprox(crit(oedlayer_wo_c)(p_wo_c, nothing), C_init, atol=1e-6)
            @test isapprox(crit(oedlayer_w_c.layer, sols_w_c), C_init, atol=1e-6)
        else
            @test isapprox(crit(oedlayer_w_c)(p_w_c, nothing), -inv(C_init), atol=1e-6)
            @test isapprox(crit(oedlayer_wo_c)(p_wo_c, nothing), -inv(C_init), atol=1e-6)
            @test isapprox(crit(oedlayer_w_c.layer, sols_w_c), -inv(C_init), atol=1e-6)
        end
    end
end

@testset "Discrete OED: fixed and with controls" begin
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

    t_measure = 0.5:1.0:11.5
    control = ControlParameter(0.0:0.01:0.99, name=:control, bounds=(-1.,1.), controls=zeros(100))
    observed = (u,p,t) -> [u[1]]
    oedlayer_wo_c = OEDLayer(prob, Tsit5(); params=[1], measurement_points = t_measure , observed=observed)
    oedlayer_w_c = OEDLayer(prob, Tsit5(); params=[1], measurement_points = t_measure, observed=observed,
                        control_indices = [2], controls=(control,))
    rng = Random.default_rng()
    ps_wo_c, st_wo_c = LuxCore.setup(rng, oedlayer_wo_c)
    ps_w_c, st_w_c = LuxCore.setup(rng, oedlayer_w_c)
    p_wo_c = ComponentArray(ps_wo_c)
    p_w_c = ComponentArray(ps_w_c)

    C_init = 43.49250925534693

    @test isapprox(Corleone.fim(oedlayer_w_c, p_w_c)[1,1], inv(C_init), atol=1e-4)
    @test isapprox(Corleone.fim(oedlayer_wo_c, p_wo_c)[1,1], inv(C_init), atol=1e-4)

    for crit in crits
        sols_w_c, _ = oedlayer_w_c(nothing, p_w_c, st_w_c)

        if crit in [ACriterion(), DCriterion(), ECriterion()]
            @test isapprox(crit(oedlayer_w_c)(p_w_c, nothing), C_init, atol=1e-4)
            @test isapprox(crit(oedlayer_wo_c)(p_wo_c, nothing), C_init, atol=1e-4)
        else
            @test isapprox(crit(oedlayer_w_c)(p_w_c, nothing), -inv(C_init), atol=1e-4)
            @test isapprox(crit(oedlayer_wo_c)(p_wo_c, nothing), -inv(C_init), atol=1e-4)
        end
    end

end