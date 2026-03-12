using Corleone
using LuxCore
using OrdinaryDiffEqTsit5
using Random
using SymbolicIndexingInterface
using Test

rng = MersenneTwister(7)

function lqr_dynamics!(du, u, p, t)
    a, b, uctrl = p
    du[1] = a * u[1] + b * uctrl
    du[2] = (u[1] - 1.0)^2 + 0.1 * uctrl^2
    return nothing
end

tspan = (0.0, 12.0)
u0 = [2.0, 0.0]
p0 = [-1.0, 1.0, 0.0]
sys = SymbolCache([:x, :cost], [:a, :b, :u], :t)
prob = ODEProblem(ODEFunction(lqr_dynamics!; sys=sys), u0, tspan, p0)

controls = (
    FixedControlParameter(name=:a, controls=(rng, t) -> [-1.0]),
    FixedControlParameter(name=:b, controls=(rng, t) -> [1.0]),
    ControlParameter(
        collect(0.0:0.1:11.9);
        name=:u,
        bounds=t -> (zero(t) .- 2.0, zero(t) .+ 2.0),
        controls=(rng, t) -> fill(0.25, length(t)),
    ),
)

@testset "Constructors and Accessors" begin
    ic = InitialCondition(prob; tunable_ic=[1], quadrature_indices=[2])
    cps = ControlParameters(controls...)

    layer_from_layers = SingleShootingLayer(ic, cps; algorithm=Tsit5(), name=:ss_lqr_1)
    layer_from_ic = SingleShootingLayer(ic, controls...; algorithm=Tsit5(), name=:ss_lqr_2)
    layer_from_prob = SingleShootingLayer(prob, controls...; algorithm=Tsit5(), name=:ss_lqr_3)

    @test layer_from_layers.name == :ss_lqr_1
    @test layer_from_ic.name == :ss_lqr_2
    @test layer_from_prob.name == :ss_lqr_3
    @test Corleone.get_problem(layer_from_prob) == prob
    @test Corleone.get_tspan(layer_from_prob) == tspan
    @test Corleone.get_quadrature_indices(layer_from_prob) == Int[]
    @test Corleone.get_tunable_u0(layer_from_layers) == [1]
    @test Corleone.get_tunable_u0(layer_from_layers, true) == [1]

    shooting_vars = Corleone.get_shooting_variables(layer_from_prob)
    @test shooting_vars.state == Int[]
    @test shooting_vars.control == [1, 2]
end

@testset "State Setup, Binning, and Evaluation" begin
    layer = SingleShootingLayer(prob, controls...; algorithm=Tsit5(), name=:ss_lqr)
    ps, st = LuxCore.setup(rng, layer)

    # 120 intervals are split into two bins because MAXBINSIZE = 100.
    @test length(st.timestops) == 2
    @test length(st.timestops[1]) == 100
    @test length(st.timestops[2]) == 20
    @test isapprox(first(st.timestops[1])[1], 0.0)
    @test isapprox(first(st.timestops[1])[2], 0.1)
    @test isapprox(last(st.timestops[2])[1], 11.9)
    @test isapprox(last(st.timestops[2])[2], 12.0)

    inputs, _ = layer.controls(st.timestops, ps.controls, st.controls)
    sols = Corleone.eval_problem(prob, layer.algorithm, true, inputs)
    @test length(sols) == 120

    traj, st2 = layer(nothing, ps, st)
    @test traj isa Corleone.Trajectory
    @test st2.system == st.system
    @test first(traj.t) == 0.0
    @test isapprox(last(traj.t), 12.0; atol=1.0e-12)
    @test all(diff(traj.t) .> 0.0)
    @test length(traj.u) == length(traj.t)
end

@testset "Symbolic Access and Default System Fallback" begin
    layer = SingleShootingLayer(prob, controls...; algorithm=Tsit5())
    ps, st = LuxCore.setup(rng, layer)
    traj = @inferred first(layer(nothing, ps, st))

    xvals = getsym(traj, :x)(traj)
    uvals = getsym(traj, :u)(traj)
    avals = getsym(traj, :a)(traj)
    bvals = getsym(traj, :b)(traj)

    @test length(xvals) == length(traj.t)
    @test length(uvals) == length(traj.t) - 1
    @test all(==(-1.0), avals)
    @test all(==(1.0), bvals)
    @test traj.ps[:u] == ps.controls[3]

    plain_prob = ODEProblem((u, p, t) -> [-0.5 * u[1] + p[1]], [1.0], (0.0, 1.0), [0.0])
    plain_control = ControlParameter([0.0, 0.5]; name=:u, controls=(rng, t) -> zeros(length(t)))
    plain_layer = SingleShootingLayer(plain_prob, plain_control; algorithm=Tsit5())
    plain_st = LuxCore.initialstates(rng, plain_layer)

    @test length(variable_symbols(plain_st.system)) == 1
    @test parameter_symbols(plain_st.system) == [:u]
end
