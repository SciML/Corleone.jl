using Corleone
using TestEnv
TestEnv.activate() 

using OrdinaryDiffEqTsit5
using Test
using Random
using LuxCore

using ComponentArrays
using Optimization, OptimizationMOI, Ipopt

using ForwardDiff
using SciMLSensitivity
using SciMLSensitivity.ReverseDiff
using SciMLSensitivity.Zygote
using SymbolicIndexingInterface

rng = Random.default_rng()

function lotka_dynamics(u, p, t)
    du1 = u[1] - p[2] * prod(u[1:2]) - 0.4 * p[1] * u[1]
    du2 = -u[2] + p[3] * prod(u[1:2]) - 0.2 * p[1] * u[2]
    du3 = (u[1] - 1.0)^2 + (u[2] - 1.0)^2
    return [du1, du2, du3]
end

tspan = (0.0, 12.0)
u0 = [0.5, 0.7, 0.0]
p0 = [0.0, 1.0, 1.0]
prob = ODEProblem(
    ODEFunction(lotka_dynamics, sys = SymbolCache([:x, :y, :c], [:fishing, :c₁, :c₂], :t)), 
    u0, tspan, p0; abstol = 1.0e-8, reltol = 1.0e-6)

control = PiecewiseConstantControl(:fishing, zero(0.0:0.1:11.9), collect(0.0:0.1:11.9), (0.0, 1.0))

layer = Corleone.SingleShootingLayer(
   prob, Tsit5(); controls = [control], 
   tunable_u0 = [:x,], 
   tunable_p = (),
   #tunable_p = [:c₁,], 
   quadrature_indices = [:c,],
   #p_bounds = ((2.0,), (2.0,)), 
   u0_bounds = ((0.0,), (1.0,))
)

ps, st = LuxCore.setup(rng, layer)
p = ComponentVector(ps) 

@inferred (first ∘ layer)(nothing, p, st)

loss = let layer = layer, st = st, ax = getaxes(p)
    (p) -> begin
        traj, _ = layer(nothing, ComponentArray(p, ax), st)
        traj[:c][end]
    end
end

p0 = collect(p)

@inferred loss(p0)

fd = ForwardDiff.gradient(loss, p0)
zyg = Zygote.gradient(loss, p0)
@test fd ≈ first(zyg)
rev = ReverseDiff.gradient(loss, p0)
@test fd ≈ rev
