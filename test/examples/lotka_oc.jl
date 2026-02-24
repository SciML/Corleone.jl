using Corleone
using TestEnv
TestEnv.activate() 

using OrdinaryDiffEqTsit5
using Test
using Random
using LuxCore

using ComponentArrays
using Optimization, OptimizationMOI, Ipopt

using SciMLSensitivity
using SciMLSensitivity.ReverseDiff
using SciMLSensitivity.Zygote
using SymbolicIndexingInterface

rng = Random.default_rng()

function lotka_dynamics!(du, u, p, t)
    du[1] = u[1] - p[2] * prod(u[1:2]) - 0.4 * p[1] * u[1]
    du[2] = -u[2] + p[3] * prod(u[1:2]) - 0.2 * p[1] * u[2]
    du[3] = (u[1] - 1.0)^2 + (u[2] - 1.0)^2
    return
end

tspan = (0.0, 12.0)
u0 = [0.5, 0.7, 0.0]
p0 = [0.0, 1.0, 1.0]
prob = ODEProblem(
    ODEFunction(lotka_dynamics!, sys = SymbolCache([:x, :y, :c], [:fishing, :c₁, :c₂], :t)), 
    u0, tspan, p0; abstol = 1.0e-8, reltol = 1.0e-6)

control = PiecewiseConstantControl(:fishing, [1., 2., 3., 4.], [1., 2., 3., 4.], (0.0, 3.0))
ps, st = LuxCore.setup(rng, control)
control(5.0, ps, st)

layer = Corleone.SingleShootingLayer(
   prob, Tsit5(); controls = [control], tunable_u0 = (:x,), tunable_p = (:c₁,), quadrature_indices = (:c,), p_bounds = ((2.0,), (2.0,)), u0_bounds = ((0.0,), (1.0,))
)

ps, st = LuxCore.setup(rng, layer)

@code_warntype layer(nothing, ps, st)

loss = let layer = layer, st = st 
    (p) -> begin
        traj, _ = layer(nothing, p, st)
        traj[:c][end]
    end
end
using ForwardDiff

ComponentVector(ps)

ForwardDiff.gradient(loss, ComponentVector(ps))

player = MultipleShootingLayer(layer, 0., 2., 3.; ensemble_algorithm = EnsembleDistributed())
ps, st = LuxCore.setup(rng, player)
player(nothing, ps, st)


a_ts = DiffEqArray(Corleone.maybevec.(ps.controls[1].u), st.controls[1].t)

ParameterTimeseriesCollection(xxx, ones(1))

### Examples 


using BenchmarkTools
@btime player(nothing, ps, st)


Corleone.clamp_tspan(layer, (0.0, 2.5)) # |> Corleone.is_shooted

Corleone.clamp_tspan(control, (2.5, 4.0))  |> Corleone.is_shooted

ps, st = LuxCore.setup(rng, layer)

layer(nothing, ps, st)

get_bounds(layer)



using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D

@parameters τ = 3.0 # parameters
@variables x(t) = 0.0  u(..) = 0.0, [input = true, bounds = (0., 1.)]# dependent variables
eqs = [D(x) ~ (1 - x) / τ + u(t)]

@named fol_model = System(eqs, t)

using OrdinaryDiffEq
fol = mtkcompile(fol_model, inputs = [u(t)])
prob = ODEProblem(fol, [], (0.0, 10.0))
control = PiecewiseConstantControl(u(t), LinRange(0., 1., 10), collect(0.0:1.0:9.0), (0.0, 1.0))
rmk = ProblemRemaker(prob, tunable_u0 = (), tunable_p = (τ, Initial(x)),)

ps, st = LuxCore.setup(rng, rmk)
ps = merge(ps, (; u0 = (), p = (2.0, 0.5)))
prob_ , _ = rmk(nothing, ps, st)



sol = solve(prob)
u0 = getsym(prob, [:x, :y])(prob)
u0 = remake_buffer(prob, prob.u0, [:x, :y], [1., 2.])

SymbolicIndexingInterface.symbolic_container(prob.f)

cgrid = collect(0.0:0.1:11.9)
N = length(cgrid)
control = ControlParameter(
    cgrid, name = :fishing, bounds = (0.0, 1.0), controls = zeros(N)
)

layer = SingleShootingLayer(prob, Tsit5(); controls = (1 => control,), bounds_p = ([1.0, 1.0], [1.0, 1.0]))

ps, st = LuxCore.setup(rng, layer)

sol, _ = layer(nothing, ps, st)

@test sol.t == getsym(sol, :t)(sol)
@test sol.p[1] == getsym(sol, :p₁)(sol)
@test sol.p[2] == getsym(sol, :p₂)(sol)

x = reduce(hcat, sol.u)

for (i, sym) in enumerate((:x₁, :x₂, :x₃, :fishing))
    getter = getsym(sol, sym)
    @test getter(sol) == x[i, :]
end

@test_nowarn @inferred layer(nothing, ps, st)

@test allunique(sol.t)
@test LuxCore.parameterlength(layer) == N + 2


for AD in (AutoForwardDiff(), AutoReverseDiff(), AutoZygote())
    prob = ODEProblem(lotka_dynamics!, u0, tspan, p0; abstol = 1.0e-8, reltol = 1.0e-6, sensealg = AD == AutoZygote() ? ForwardDiffSensitivity() : SciMLBase.NoAD())

    cgrid = collect(0.0:0.1:11.9)
    N = length(cgrid)
    control = ControlParameter(
        cgrid, name = :fishing, bounds = (0.0, 1.0), controls = zeros(N)
    )

    layer = SingleShootingLayer(prob, Tsit5(); controls = (1 => control,), bounds_p = ([1.0, 1.0], [1.0, 1.0]))

    ps, st = LuxCore.setup(rng, layer)

    p = ComponentArray(ps)
    lb, ub = Corleone.get_bounds(layer)

    @test lb.p == ub.p == p0[2:end]
    @test lb.controls == zeros(N)
    @test ub.controls == ones(N)
    @test size(p, 1) == LuxCore.parameterlength(layer)

    optprob = OptimizationProblem(layer, AD, Val(:ComponentArrays), loss = :x₃)

    @test isapprox(optprob.f(optprob.u0, optprob.p), 6.062277454291031, atol = 1.0e-4)

    sol = solve(
        optprob, Ipopt.Optimizer(), max_iter = 1000, tol = 5.0e-6,
        hessian_approximation = "limited-memory"
    )

    @test SciMLBase.successful_retcode(sol)
    @test isapprox(sol.objective, 1.344336, atol = 1.0e-4)

    p_opt = sol.u .+ zero(p)

    @test isempty(p_opt.u0)
    @test p_opt.p == p0[2:end]
end
