using Corleone
using OrdinaryDiffEqTsit5
using OrdinaryDiffEqBDF
using Test
using Random
using LuxCore
using Symbolics
using ForwardDiff
using LinearAlgebra

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

@testset "Dimensions" begin
    @test Corleone.is_fixed(oedlayer_wo_c) == true
    @test Corleone.is_fixed(oedlayer_w_c) == false
    @test oedlayer_w_c.dimensions.np == oedlayer_wo_c.dimensions.np == 2
    @test oedlayer_w_c.dimensions.np_fisher == oedlayer_wo_c.dimensions.np_fisher == 1
    @test oedlayer_w_c.dimensions.nx == oedlayer_wo_c.dimensions.nx == 2
    @test oedlayer_w_c.dimensions.nh == oedlayer_wo_c.dimensions.nh == 1
    @test oedlayer_w_c.dimensions.nc == 1
    @test oedlayer_wo_c.dimensions.nc == 0
end

@testset "Bounds" begin
    lb_wo_c, ub_wo_c = Corleone.get_bounds(oedlayer_wo_c)
    lb_w_c, ub_w_c = Corleone.get_bounds(oedlayer_w_c)

    @test length(lb_wo_c.controls) == length(ub_wo_c.controls) == 10
    @test length(lb_w_c.controls) == length(ub_w_c.controls) == 110
    @test all(lb_wo_c.controls .== 0.0)
    @test all(ub_wo_c.controls .== 1.0)
    @test all(lb_w_c.controls[1:100] .== -1.0)
    @test all(lb_w_c.controls[101:end] .== 0.0)
end

@testset "Predictions" begin
    rng = Random.default_rng()
    params_wo_c = LuxCore.setup(rng, oedlayer_wo_c)
    params_w_c = LuxCore.setup(rng, oedlayer_w_c)
    sol_wo_c, _ = oedlayer_wo_c(nothing, params_wo_c...)
    sol_w_c, _ = oedlayer_w_c(nothing, params_w_c...)
    @test isapprox(sol.u[end], sol_wo_c.u[end][1:2], atol=1e-7)
    @test isapprox(sol.u[end], sol_w_c.u[end][1:2] , atol=1e-7)
    @test isapprox(sol_w_c.u[end][3:5], sol_wo_c.u[end][3:5], atol=1e-7)
end

@testset "Fisher variables and symmetric_from_vector" begin
    f_sym_wo_c = Corleone.fisher_variables(oedlayer_wo_c)
    f_sym_w_c = Corleone.fisher_variables(oedlayer_w_c)
    @test isnothing(f_sym_wo_c) # All fixed -> this is covered by Corleone.observed_sensitivity_product_variables
    @test length(f_sym_w_c) == 1

    # Need a bigger system to test whether setup of FIM works as expected
    simple_dyn = (u,p,t) -> [prod(vcat(u[1], p)); (u[1]-1.0)^2]

    for i=1:5
        _p = vcat(ones(i), 0.0)
        _prob = ODEProblem(simple_dyn, u0, tspan, _p)
        oedlayer = OEDLayer(_prob, Tsit5(); observed=observed, params=1:i,
                            control_indices = [i+1], controls=(control,))

        fsym = Corleone.fisher_variables(oedlayer)
        @test length(fsym) == i*(i+1)/2

        F = Corleone.symmetric_from_vector(fsym)
        F_symbols = map(Iterators.product(1:i, 1:i)) do (i,j)
            if i<=j
                string(Symbol("F", join(Symbolics.map_subscripts.([i,j]), "ˏ")))
            else
                string(Symbol("F", join(Symbolics.map_subscripts.([j,i]), "ˏ")))
            end
        end
        @test all([string(F[i,j]) == F_symbols[i,j] for i in 1:i for j in 1:i])
    end

end

@testset "DAE augmentation" begin
    T₀ = 69 + 273.15
    R = 1.987204258640
    Q = 0.0131

    function dow(du, u, p, t)

        y₁,y₂,y₃,y₄,y₅,y₆,y₇ ,y₈ ,y₉ ,y₁₀ = u
        dy₁,dy₂,dy₃,dy₄,dy₅,dy₆,dy₇ ,dy₈ ,dy₉ ,dy₁₀ = du
        temperature = p[10]
        k₁  = exp(p[1]) * exp(-p[4] * 1.e4/(R) * (1/temperature - 1/T₀))
        k₂  = exp(p[2]) * exp(-p[5] * 1.e4/(R) * (1/temperature - 1/T₀))
        k₋₁ = exp(p[3]) * exp(-p[6] * 1.e4/(R) * (1/temperature - 1/T₀))

        # abbreviation of ODE
        f₁ = -k₂ * y₈ * y₂
        f₂ = -k₁ * y₆ * y₂ + k₋₁ * y₁₀ - k₂ * y₈ * y₂
        f₃ = k₂ * y₈ * y₂ + k₁ * y₆ * y₄ - 0.5 * k₋₁ * y₉
        f₄ = -k₁ * y₆ * y₄ + 0.5 * k₋₁ * y₉
        f₅ = k₁ * y₆ * y₂ - k₋₁ * y₁₀
        f₆ = -k₁ * (y₆ * y₂ + y₆ * y₄) + k₋₁ * (y₁₀ + 0.5 * y₉)

        return [
        -dy₁ +  f₁;
        -dy₂ +  f₂;
        -dy₃ +  f₃;
        -dy₄ +  f₄;
        -dy₅ +  f₅;
        -dy₆ +  f₆;
            - y₇ - Q + y₆ + y₈ + y₉ + y₁₀;
            - y₈ + (exp(p[8]) * y₁) / (exp(p[8]) + y₇);
            - y₉ + (exp(p[9]) * y₃) / (exp(p[9]) + y₇);
            - y₁₀ + (exp(p[7]) * y₅) / (exp(p[7]) + y₇)
        ]
    end

    p = [0.8010374972073442, 1.1069954919870142, 27.29000930653549, 1.847, 1.882, 2.636, -38.7599775331355, -14.260002398041287, -39.14394658089878]
    u0 = [1.7066, 8.32, 0.01, 0.0, 0.0, 0.0131, 0.0010457132164084471, 0.0010457132164084471, 0.0, 0.0]
    tspan = (0.,200.0)
    du0 = zeros(10)

    prob = DAEProblem(dow, du0, u0, tspan, vcat(p, 40.0 + 273.15), abstol=1e-8, reltol=1e-6)

    oedlayer = OEDLayer(prob, DFBDF(); params =1:9, observed = (u,p,t) -> u[1:4])
    ps, st = LuxCore.setup(rng, oedlayer)

    sols, _ = oedlayer(nothing, ps, st)
    sensitivities = Corleone.sensitivity_variables(oedlayer)
    idx_t10 = findfirst(x -> x ==10.0, sols.t)
    G_aug = sols[sensitivities][idx_t10]
    pred_(p) = begin
        Array(solve(remake(prob, p=vcat(p, 40.0+273.15)), DFBDF(), saveat=[10.0]))
    end

    G_fd = ForwardDiff.jacobian(pred_, p)

    @test norm(G_aug .- G_fd, Inf) < 1e-3
end