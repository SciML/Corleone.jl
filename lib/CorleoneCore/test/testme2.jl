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

CorleoneCore.restrict_controls(control, 0.0, 3.0)
CorleoneCore.restrict_controls((control,), 0.0, 3.0)
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
mssol, _ = mslayer(nothing, ComponentArray(ms_ps), ms_st);

f = Figure()
ax = CairoMakie.Axis(f[1,1])
[plot!(ax,sol.time, sol.states[i,:],color=:grey) for sol in mssol for i in 1:3]
f

ps = ComponentArray(ms_ps)

loss = let mslayer = mslayer, st = ms_st, ax = getaxes(ps)
    (p, ::Any) -> begin
        ps = ComponentArray(p, ax)
        sols, _ = mslayer(nothing, ps, st)
        last(sols).states[end,end]
    end
end

p = ComponentArray(ms_ps)
loss(p, nothing)

loss(collect(p), nothing)

grad_fd = ForwardDiff.gradient(Base.Fix2(loss, nothing), collect(p))

grad_zg = Zygote.gradient(Base.Fix2(loss, nothing), collect(p))[1]


shooting_constraints = let mslayer = mslayer, st = ms_st, ax = getaxes(ps)
    (p, ::Any) -> begin
        ps = ComponentArray(p, ax)
        sols, _ = mslayer(nothing, ps, st)
        reduce(vcat, map(zip(sols[1:end-1], keys(ax[1])[2:end])) do (sol, name_i)
            _u0 = getproperty(ps, name_i).u0
            sol.states[:,end] .-_u0
        end)
    end
end

ll = shooting_constraints(p, nothing)

eq_cons(res, x, p) = res .= shooting_constraints(x, p)

jac_fd = ForwardDiff.jacobian(Base.Fix2(shooting_constraints, nothing), collect(p))
spy(jac_fd)

lb, ub = copy(p), copy(p)
lb.layer_2.u0 = zeros(3) .+ 0.05
lb.layer_3.u0 = zeros(3) .+ 0.05
lb.layer_4.u0 = zeros(3) .+ 0.05

ub.layer_2.u0 = zeros(3) .+ 10.0
ub.layer_3.u0 = zeros(3) .+ 10.0
ub.layer_4.u0 = zeros(3) .+ 10.0
ub.layer_1.controls .= 1.0
ub.layer_2.controls .= 1.0
ub.layer_3.controls .= 1.0
ub.layer_4.controls .= 1.0


fig_grad = scatter(grad_fd)
scatter!(grad_zg)
display(fig_grad)

optfun = OptimizationFunction(
    loss, AutoForwardDiff(), cons = eq_cons
)

optprob = OptimizationProblem(
    optfun, collect(p), lb = collect(lb), ub = collect(ub), lcons = zero(ll), ucons=zero(ll)
)



uopt = solve(optprob, Ipopt.Optimizer(),
     tol = 1e-6,
     hessian_approximation = "limited-memory",
     max_iter = 300 # 165
)

blocks = begin
    l = map(x -> length(ComponentArray(x)), ms_ps)
    lc = cumsum(ComponentArray(l)[:])
    vcat(0, lc)
end

uopt = solve(optprob, BlockSQPOpt(),
    opttol = 1e-6,
    options = blockSQP.sparse_options(),
    sparsity = blocks,
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