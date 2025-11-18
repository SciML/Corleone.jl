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
layer = MultipleShootingLayer(prob, Tsit5(), shooting_points,  [1], (control,), shooting_points)
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

#=
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


@testset "Construction and initialization of OEDLayer / MultiExperimentLayer" begin
    oed_ms_layer = @test_nowarn OEDLayer(layer, observed = (u,p,t) -> u[1:2],
                                    params = [2,3], dt=0.25);
    oed_multiexperiment = @test_nowarn MultiExperimentLayer(oed_ms_layer, 3)

    ps_ms, st_ms = LuxCore.setup(rng, oed_ms_layer)
    ps_multi, st_multi = LuxCore.setup(rng, oed_multiexperiment)
    ps_def, st_def = @test_nowarn DefaultsInitialization()(rng, oed_ms_layer)
    ps_def_multi, st_def = @test_nowarn DefaultsInitialization()(rng, oed_multiexperiment)
    @test ps_def == ps_ms
    @test all([getproperty(ps_def_multi, Symbol("experiment_$i")) == ps_def for i=1:3])


    # Testing dimensions
    @test length(ps_ms) == length(shooting_points)
    @test length(ps_multi) == 3
    @test length(ps_multi.experiment_1) == length(ps_ms)
    @test ps_multi.experiment_1 == ps_ms
    dims = oed_ms_layer.dimensions
    aug_u0 = first(oed_ms_layer.layer.layers).problem.u0
    nx_augmented = dims.nx + dims.nx*dims.np_fisher + (dims.np_fisher+1)*dims.np_fisher/2 |> Int
    @test length(aug_u0) == nx_augmented

    # Testing criteria
    crit = ACriterion()
    ACrit_single = crit(oed_ms_layer)
    ACrit_multi = crit(oed_multiexperiment)
    @test isapprox(ACrit_single(ComponentArray(ps_def), nothing), 3 * ACrit_multi(ComponentArray(ps_def_multi), nothing))

    # Testing block structures
    block_structure_ms = Corleone.get_block_structure(oed_ms_layer)
    block_structure_multi_calculated = vcat(block_structure_ms, block_structure_ms[2:end] .+ last(block_structure_ms))
    block_structure_multi_calculated = vcat(block_structure_multi_calculated, block_structure_ms[2:end] .+ last(block_structure_multi_calculated))
    block_structure_multi_evaluated = Corleone.get_block_structure(oed_multiexperiment)
    @test block_structure_multi_calculated == block_structure_multi_evaluated

    # Testing initializations
    fwd_init = ForwardSolveInitialization()
    lin_init = LinearInterpolationInitialization(Dict(1:nx_augmented .=> 2.0))
    const_init = ConstantInitialization(Dict(1:nx_augmented .=> 1.0))
    rands = [rand(3) for i=1:nx_augmented]
    custom_init = CustomInitialization(Dict(1:nx_augmented .=> map(i -> vcat(aug_u0[i], rands[i]), 1:nx_augmented)))
    hybrid_init = HybridInitialization(Dict(1 => const_init,
                                            2 => custom_init,
                                            3 => lin_init), fwd_init)

    testhybrid(p) = begin
        inits = vcat(
        p.layer_2.u0[1:3] == [1.0, rands[2][1], aug_u0[3] + 3/12 * (2.0 - aug_u0[3])],
        p.layer_3.u0[1:3] == [1.0, rands[2][2], aug_u0[3] + 6/12 * (2.0 - aug_u0[3])]
        )
    end
    testlin(p) = all([isapprox(getproperty(p, Symbol("layer_$i")).u0, aug_u0 + (2.0 .- aug_u0) * 3*(i-1)/12, atol=1e-4) for i=2:length(shooting_points)])
    testconst(p) = all([isapprox(getproperty(p, Symbol("layer_$i")).u0, ones(nx_augmented), atol=1e-8) for i=2:length(shooting_points)])
    testcustom(p) = all([isapprox(getproperty(p, Symbol("layer_$i")).u0, reduce(vcat, [x[i-1] for x in rands]), atol=1e-8) for i=2:length(shooting_points)])

    for _layer in [oed_ms_layer, oed_multiexperiment]
        for (init, test) in zip([hybrid_init, fwd_init, lin_init, const_init, custom_init], [testhybrid, nothing, testlin, testconst, testcustom])
            ps_init, st_init = init(rng, _layer)
            if init in [fwd_init, hybrid_init]
                shooting_constraints = Corleone.get_shooting_constraints(_layer)
                sols_fwd, _ = _layer(nothing, ps_init, st_init)
                if init == fwd_init
                    @test norm(shooting_constraints(sols_fwd, ps_init)) < 1e-8
                else
                    eval_shooting = shooting_constraints(sols_fwd, ps_init)
                    indices_fwd = trues(length(eval_shooting))
                    indices_fwd[1:nx_augmented:end] .= false
                    indices_fwd[2:nx_augmented:end] .= false
                    indices_fwd[3:nx_augmented:end] .= false
                    @test norm(shooting_constraints(sols_fwd, ps_init)[indices_fwd]) < 1e-8
                end
            else
                if _layer == oed_ms_layer
                    @test test(ps_init)
                else
                    @test all([test(getproperty(ps_init, Symbol("experiment_$i"))) for i=1:3])
                end
            end
        end
    end
end

=#
