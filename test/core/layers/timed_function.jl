using TestEnv
using Corleone
TestEnv.activate()
using Test

using Corleone
using Corleone: Solutions
using OrdinaryDiffEqTsit5
using StableRNGs
using LuxCore
using SymbolicIndexingInterface
using SciMLBase
using Random

include(joinpath(@__FILE__, "..", "..", "..", "helper.jl"))
rng = Random.default_rng()
prob = ControlledLotka.generate()
cgrid = collect(LinRange(0.0, 12.0, 6))
pc1 = PiecewiseParameter(:u1, [2.0, 4.0, 6.0, 8.0, 10.0])
pc2 = PiecewiseParameter(:u2, [3.0, 10.0, 11.0])
prob = remake(prob, saveat = collect(0.0:3.0:12.0))
# Symbol[] → no tunable ICs; fixed initial condition from the problem
layer = ShootingLayer(prob, Symbol[], pc1, pc2; quadratures = [:L], shooting_method = AutoBlock(5), algorithm = Tsit5())
ps, st = LuxCore.setup(rng, layer)

traj = first(layer(remake(prob, saveat = 0.25), ps, st))
