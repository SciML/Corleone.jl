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
            -u[2] + p[3] * prod(u[1:2]) - 0.2 * p[1] * u[2];
            (u[1]-1.0)^2 + (u[2] - 1.0)^2]
end

tspan = (0., 12.)
u0 = [0.5, 0.7, 0.]
p0 = [0.0, 1.0, 1.0]

prob = ODEProblem(lotka_dynamics, u0, tspan, p0; abstol=1e-8, reltol=1e-6)

cgrid = collect(0.0:0.1:11.9)
N = length(cgrid)
control = ControlParameter(
    cgrid, name = :fishing, bounds=(0.0,1.0), controls = zeros(N)
)

# Multiple Shooting
shooting_points = [0.0, 3.0, 6.0, 9.0]
layer = MultipleShootingLayer(prob, Tsit5(), [1], (control,), shooting_points)

Ni = N / length(shooting_points) |> Int
np_without_controls = length(setdiff(eachindex(prob.p), layer.layers[1].control_indices))
nx = length(prob.u0)
ps, st = LuxCore.setup(rng, layer)
p = ComponentArray(ps)
lb, ub = Corleone.get_bounds(layer)

@testset "General Multiple shooting tests" begin
    @test Corleone.is_fixed(layer) == false
    @test length(ps) == 4 # shooting stages
    @test isempty(ps.layer_1.u0) # initial condition is not tunable
    @test all([length(getproperty(p, Symbol("layer_$i")).u0) == 3 for i=2:4]) # ICs of subsequent layers are tunable

    blocks = cumsum(vcat(0, Ni+np_without_controls, [Ni+np_without_controls+nx for _ = 2:4]))
    @test Corleone.get_block_structure(layer) == blocks
end

@testset "Initialization methods" begin
    # ForwardSolve
    sol_at_shooting_points = solve(prob, Tsit5(), saveat=shooting_points)
    fwd_init = ForwardSolveInitialization()
    ps_fwd, _ = fwd_init(rng, layer)
    @test isempty(ps_fwd.layer_1.u0)
    @test all([isapprox(sol_at_shooting_points[i], getproperty(ps_fwd, Symbol("layer_$i")).u0, atol=1e-5) for i=2:4])
    matching_constraints = Corleone.get_shooting_constraints(layer)
    sol_fwd, _ = layer(nothing, ps_fwd, st)
    @test norm(matching_constraints(sol_fwd, ps_fwd)) < 1e-8

    # ConstantInitialization
    const_init = ConstantInitialization(Dict(1 => 0.9, 2 => 1.1, 3 => 0.0))
    ps_const, _ = const_init(rng, layer)
    @test isempty(ps_const.layer_1.u0)
    @test all([getproperty(ps_const, Symbol("layer_$i")).u0 == [0.9, 1.1, 0.0] for i=2:4])

    # DefaultsInitialization
    def_init = DefaultsInitialization()
    ps_def, _ = def_init(rng, layer)
    @test isempty(ps_def.layer_1.u0)
    @test all([getproperty(ps_def, Symbol("layer_$i")).u0 == u0 for i=2:4])

    # LinearInterpolationInitialization
    lin_init = LinearInterpolationInitialization(Dict(1=> 2.0, 2=>1.0, 3=> 1.34))
    ps_lin, _ = lin_init(rng, layer)
    @test isempty(ps_lin.layer_1.u0)
    @test isapprox(ps_lin.layer_2.u0, u0 .+ ([2.0, 1.0, 1.34] .- u0) * 3/12, atol=1e-7)
    @test isapprox(ps_lin.layer_3.u0, u0 .+ ([2.0, 1.0, 1.34] .- u0) * 6/12, atol=1e-7)
    @test isapprox(ps_lin.layer_4.u0, u0 .+ ([2.0, 1.0, 1.34] .- u0) * 9/12, atol=1e-7)

    custom_init = CustomInitialization(Dict(1 => vcat(u0[1], ones(3)),
                                            2 => vcat(u0[2], 1.1*ones(3)),
                                            3 => vcat(u0[3], 1.34 * ones(3))))
    ps_custom, _ = custom_init(rng, layer)
    @test isempty(ps_custom.layer_1.u0)
    @test all([getproperty(ps_custom, Symbol("layer_$i")).u0 == [1.0, 1.1, 1.34] for i=2:4])

    # Hybrid initialization
    hybrid_init = HybridInitialization(Dict(1 => lin_init,
                                            2 => custom_init), ForwardSolveInitialization())
    ps_hybrid, _ = hybrid_init(rng, layer)

    @test isempty(ps_hybrid.layer_1.u0)
    @test ps_hybrid.layer_2.u0[1:2] == [u0[1] + (2.0-u0[1]) * 3/12, 1.1]
    @test ps_hybrid.layer_3.u0[1:2] == [u0[1] + (2.0-u0[1]) * 6/12, 1.1]
    @test ps_hybrid.layer_4.u0[1:2] == [u0[1] + (2.0-u0[1]) * 9/12, 1.1]

    sol_hybrid, _ = layer(nothing, ps_hybrid, st)
    matching_hybrid = matching_constraints(sol_hybrid, ps_hybrid)
    @test norm(matching_hybrid[3:3:end]) < 1e-9
end