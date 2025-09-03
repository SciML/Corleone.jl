using Pkg
Pkg.activate(joinpath(pwd(), "lib/CorleoneCore"))

using CorleoneCore
using OrdinaryDiffEq
using SciMLSensitivity
using ComponentArrays
using LuxCore
using Random

using CairoMakie
using BenchmarkTools
using Zygote
using ForwardDiff

using Optimization
using OptimizationMOI
using Ipopt
using blockSQP

function lotka_dynamics(du, u, p, t)
    du[1] = u[1] - prod(u[1:2]) - 0.4 * p[1] * u[1]
    du[2] = -u[2] + prod(u[1:2]) - 0.2 * p[1] * u[2]
    du[3] = (u[1]-1.0)^2 + (u[2] - 1.0)^2
end

tspan = (0., 12.)
u0 = [0.5, 0.7, 0.]
p0 = [0.0]
prob = ODEProblem(lotka_dynamics, u0, tspan, p0,
    abstol = 1e-8, reltol = 1e-6, sensealg = ForwardDiffSensitivity()
    )
control = ControlParameter(
    0.0:0.1:11.9, name = :fishing
)
layer = CorleoneCore.SingleShootingLayer(prob, Tsit5(),Int64[],[1], (control,))
ps, st = LuxCore.setup(Random.default_rng(), layer)
p = ComponentArray(ps)

sprob = CorleoneCore.SingleShootingProblem(layer, ps, st)
ssol, _ = solve(sprob, CorleoneCore.DummySolve());

f = Figure()
ax = CairoMakie.Axis(f[1,1])
[plot!(ax,ssol.time, ssol.states[i,:]) for i in 1:size(ssol.states,1)]
f


shooting_points = [0.0, 3.0, 6.0, 9.0, 12.0] # cost
mslayer = CorleoneCore.MultipleShootingLayer(prob, Tsit5(), Int64[], [1], (control,), shooting_points);

ms_ps, ms_st = LuxCore.setup(Random.default_rng(), mslayer)
mssol, _ = mslayer(nothing, ms_ps, ms_st);

f = Figure()
ax = CairoMakie.Axis(f[1,1])
[plot!(ax,sol[1].time, sol[1].states[i,:],color=:grey) for sol in mssol for i in 1:3]
f


loss = let layer = mslayer, st = ms_st, ax = getaxes(ms_ps)
    (p, ::Any) -> begin
        ps = ComponentArray(p, ax)
        sols, _ = layer(nothing, ps, st)
        Array(sols[end])[3, end]
    end
end

loss(p, nothing)

loss(collect(p), nothing)

grad_fd = ForwardDiff.gradient(Base.Fix2(loss, nothing), collect(p))

#grad_zg = Zygote.gradient(Base.Fix2(loss, nothing), collect(p))[1]

fig_grad = scatter(grad_fd)
scatter!(grad_zg)
display(fig_grad)

optfun = OptimizationFunction(
    loss, AutoForwardDiff()
)

optprob = OptimizationProblem(
    optfun, collect(p), lb = zero(collect(p)), ub = zero(collect(p)) .+ 1
)

uopt = solve(optprob, Ipopt.Optimizer(),
     tol = 1e-6,
     #hessian_approximation = "limited-memory",
     max_iter = 300 # 165
)

uopt = solve(optprob, BlockSQPOpt(),
     opttol = 1e-6,
     #hessian_approximation = "limited-memory",
     maxiters = 300 # 165
)

t = LinRange(0., 11.9, length(uopt))
fcontrols = stairs(t, uopt)
display(fcontrols)

sol, st = layer(prob, uopt .+ zero(p), st)
sol0, st = simlayer(prob, p, st)
fsol = plot(sol)


uopt = solve(optprob, Ipopt.Optimizer(),
     #tol = 1e-6,
     #hessian_approximation = "limited-memory",
     max_iter = 200 # 165
)

popt = ComponentArray(uopt) + zero(p)
nopt = NamedTuple(popt)
u1 = nopt.problems[1].p.controls.layers.layer_1.local_controls
#u2 = nopt.problems[1].p.controls.layers.layer_2.local_controls
t = LinRange(t0, tinf-Δt, length(u1))
fcontrols = stairs(t, u1)
#stairs!(t, u2)
display(fcontrols)


sol, st = simlayer(problem, popt .+ zero(p), st)
sol0, st = simlayer(problem, p, st)
fsol = plot(sol)