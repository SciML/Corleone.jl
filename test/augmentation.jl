using Corleone
using OrdinaryDiffEqTsit5
using Test
using Random
using LuxCore
using Symbolics

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
oedlayer_wo_c = OEDLayer(prob, Tsit5(); params=[1], observed=observed)
oedlayer_w_c = OEDLayer(prob, Tsit5(); params=[1], observed=observed,
                    control_indices = [2], controls=(control,))

@testset "Dimensions" begin
    @test oedlayer_w_c.dimensions.np == oedlayer_wo_c.dimensions.np == 2
    @test oedlayer_w_c.dimensions.np_fisher == oedlayer_wo_c.dimensions.np_fisher == 1
    @test oedlayer_w_c.dimensions.nx == oedlayer_wo_c.dimensions.nx == 2
    @test oedlayer_w_c.dimensions.nh == oedlayer_wo_c.dimensions.nh == 1
    @test oedlayer_w_c.dimensions.nc == 1
    @test oedlayer_wo_c.dimensions.nc == 0
end

@testset "Predictions" begin
    rng = Random.default_rng()
    params_wo_c = LuxCore.setup(rng, oedlayer_wo_c)
    params_w_c = LuxCore.setup(rng, oedlayer_w_c)
    sol_wo_c, _ = oedlayer_wo_c(nothing, params_wo_c...)
    sol_w_c, _ = oedlayer_w_c(nothing, params_w_c...)
    @test isapprox(sol[end], sol_wo_c.u[end][1:2], atol=1e-7)
    @test isapprox(sol[end], sol_w_c.u[end][1:2] , atol=1e-7)
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