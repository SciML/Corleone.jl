using Pkg
Pkg.activate(@__DIR__)
using Corleone
using CorleoneOED
using OrdinaryDiffEq
using SciMLSensitivity
using ComponentArrays
using LuxCore
using Random

using CairoMakie
using Optimization
using OptimizationMOI
using Ipopt
#using blockSQP
using LinearAlgebra


function compartmental(u, p, t)
    θ₁, θ₂, θ₃ = p
    return [θ₃ * (-θ₁ * exp(-θ₁ * t) + θ₂ * exp(-θ₂ * t));]
end

tspan = (0., 50.0)
u0 = [0.]
p0 = [0.05884, 4.298, 21.80]
prob = ODEProblem(compartmental, u0, tspan, p0,
    )

plot(solve(prob, Tsit5()))

measurement_points = [0.0, 0.166, 0.333, 0.5, 0.666, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 24.0, 30.0, 48.0]
w_init = 0.5 * ones(length(measurement_points))

oed = OEDLayer{true}(
  prob, Tsit5(),
  bounds_p = (p0, p0),
  params=[1,2,3],
  measurements=[
    ControlParameter(measurement_points, controls = w_init, bounds = (0.0, 1.0)),
  ],
  observed=(u, p, t) -> u[1:1],
)

ps, st = LuxCore.setup(Random.default_rng(), oed)

sol, _ = oed(nothing, ps, st)

function plot_oed(sol, sampling)
    f = Figure()
    ax = CairoMakie.Axis(f[1, 1],  xticks = 0:10:50, title = "States")
    ax2 = CairoMakie.Axis(f[2, 1], xticks = 0:10:50, title = "Sensitivities")
    ax3 = CairoMakie.Axis(f[3, 1], xticks = 0:10:50, title = "Sampling")
    plot!(ax, sol, idxs = [1])
    plot!(ax2, sol, idxs = [2, 3, 4])
    scatter!(ax3, measurement_points, sampling)
    f
end

plot_oed(sol, ps.controls)

optprob = OptimizationProblem(oed, ACriterion(); M =[6.0])

uopt = solve(optprob, Ipopt.Optimizer(),
    tol=1e-12,
    hessian_approximation="limited-memory",
    max_iter=100,
)

optu = uopt.u + zero(ComponentArray(ps))
sol, _  = oed(nothing, optu, st)

plot_oed(sol, optu.controls)