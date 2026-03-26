using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using TestEnv
TestEnv.activate()

using Corleone
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using OrdinaryDiffEqTsit5;
using Random;
using LuxCore;

rng = Random.default_rng()

@variables begin
    x(t) = 1.0, [tunable = false, bounds = (0.0, 1.0)]
    u(t) = 1.0, [input = true, bounds = (0.0, 1.0)]
end
@parameters begin
    p = 1.0, [bounds = (-1.0, 1.0)]
end
eqs = [D(x) ~ p * x - u]
@named simple = ODESystem(eqs, t)
layer = SingleShootingLayer(simple, [], u => 0.0:0.1:1.0, algorithm = Tsit5(), tspan = (0.0, 1.0));
ps, st = LuxCore.setup(rng, layer);

traj, st = layer(nothing, ps, st);

# This is not right
traj.ps[u]

using SymbolicIndexingInterface

any(Base.isequal(u), parameter_symbols(traj))

traj.sys

using CairoMakie

plot(traj)
