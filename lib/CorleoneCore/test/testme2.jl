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
    du[1] = u[1] - p[2] * prod(u[1:2]) - 0.4 * p[1] * u[1]
    du[2] = -u[2] + p[3] * prod(u[1:2]) - 0.2 * p[1] * u[2]
    du[3] = (u[1]-1.0)^2 + (u[2] - 1.0)^2
end

tspan = (0., 12.)
u0 = [0.5, 0.7, 0.]
p0 = [0.0, 1.0, 1.0]
prob = ODEProblem(lotka_dynamics, u0, tspan, p0,
    abstol = 1e-8, reltol = 1e-6, sensealg = ForwardDiffSensitivity()
    )
control = ControlParameter(
    0.0:0.1:11.9, name = :fishing
)
w1 = ControlParameter(
    0.0:0.1:11.9, name = :w1, controls = ones(48)
)
w2 = ControlParameter(
    0.0:0.1:11.9, name = :w2, controls=ones(48)
)
layer = CorleoneCore.SingleShootingLayer(prob, Tsit5(),Int64[],[1], (control,))
ps, st = LuxCore.setup(Random.default_rng(), layer)
p = ComponentArray(ps)

oedprob = CorleoneCore.augment_dynamics_for_oed(layer; observed = (u,p,t) -> u[1:2])
psol = solve(oedprob, Tsit5())
plot(psol)

sprob = CorleoneCore.SingleShootingProblem(layer, ps, st)
ssol, _ = solve(sprob, CorleoneCore.DummySolve());

f = Figure()
ax = CairoMakie.Axis(f[1,1])
[plot!(ax,ssol.time, ssol.states[i,:]) for i in 1:size(ssol.states,1)]
f


shooting_points = [0.0, 3.0, 6.0, 9.0, 12.0] # cost
mslayer = CorleoneCore.MultipleShootingLayer(prob, Tsit5(), Int64[], [1], (control,), shooting_points);
rng = Random.default_rng()
ms_ps, ms_st = LuxCore.setup(rng, mslayer)
ms_ps_def, _ = CorleoneCore.DefaultsInitialization()(rng, mslayer)
ms_ps_rand, _ = CorleoneCore.RandomInitialization()(rng, mslayer)
ms_ps_fwd, _ = CorleoneCore.ForwardSolveInitialization()(rng, mslayer)
lin_init = CorleoneCore.LinearInterpolationInitialization(Dict(1 => 1.0, 2 => 1.0, 3 => 1.34))
ms_ps_lin, _ = lin_init(rng, mslayer)

custom_init = CorleoneCore.CustomInitialization(Dict(1 => ones(4), 2 => .5 * ones(4), 3=> rand(4)))
ms_ps_custom, _ = custom_init(rng, mslayer)

const_init = CorleoneCore.ConstantInitialization(Dict(1 => 1.0, 2 => 2.0, 3 => 1.34))
ms_ps_const, _ = const_init(rng, mslayer)


inits = Dict([1] => CorleoneCore.LinearInterpolationInitialization(Dict(1 => 1.0)),
             [2] => CorleoneCore.CustomInitialization(Dict(2 => [0.7; 1.3; 0.9; 1.1])))

hybrid_init = CorleoneCore.HybridInitialization(inits, CorleoneCore.ForwardSolveInitialization())
ms_ps_hybrid, _ = hybrid_init(rng, mslayer)

mssol, _ = mslayer(nothing, ms_ps_hybrid, ms_st);
f = Figure()
ax1 = CairoMakie.Axis(f[1,1])
[plot!(ax1,sol.time, sol.states[i,:],color=:grey) for sol in mssol for i=1:size(sol.states,1)]
f


oed_mslayer = CorleoneCore.MultipleShootingLayer(oedprob, Tsit5(), Int64[], [1,4,5], (control,w1,w2), shooting_points);

oedprob = CorleoneCore.augment_dynamics_for_oed(mslayer; observed = (u,p,t) -> u[1:2])

psol = solve(oedprob, Tsit5())
plot(psol)
oed_ss = CorleoneCore.SingleShootingLayer(oedprob, Tsit5(), Int64[], [1,4,5], (control,w1,w2))


oed_ss = CorleoneCore.augment_layer_for_oed(layer; observed = (u,p,t) -> u[1:2])
oed_mslayer1 = CorleoneCore.augment_layer_for_oed(mslayer; observed = (u,p,t) -> u[1:2])

oed_ps, oed_st = LuxCore.setup(Random.default_rng(), oed_ss)

ms_ps1, ms_st1 = LuxCore.setup(Random.default_rng(), oed_mslayer1)
mssol, _ = oed_mslayer(nothing, ms_ps, ms_st);
mssol1, _ = oed_mslayer1(nothing, ms_ps1, ms_st1);

f = Figure()
ax = CairoMakie.Axis(f[1,1])
ax1 = CairoMakie.Axis(f[2,1])
[plot!(ax,sol.time, sol.states[i,:],color=:grey) for sol in mssol for i=1:size(sol.states,1)]
[plot!(ax1,sol.time, sol.states[i,:],color=:grey) for sol in mssol1 for i=1:size(sol.states,1)]
f

ps = ComponentArray(ms_ps)
ps = ComponentArray(oed_ps)
using LinearAlgebra
loss = let mslayer = oed_mslayer, st = ms_st, ax = getaxes(ps)
    (p, ::Any) -> begin
        ps = ComponentArray(p, ax)
        sols, _ = mslayer(nothing, ps, st)
        #inv(tr())
        inv(tr(reshape(last(sols).states[end-3:end,end], (2,2))))
    end
end

p = ComponentArray(ms_ps)
loss(ps, nothing)

loss(collect(p), nothing)

grad_fd = ForwardDiff.gradient(Base.Fix2(loss, nothing), collect(p))

grad_zg = Zygote.gradient(Base.Fix2(loss, nothing), collect(p))[1]


shooting_constraints = let mslayer = oed_mslayer, st = ms_st, ax = getaxes(ps)
    (p, ::Any) -> begin
        ps = ComponentArray(p, ax)
        sols, _ = mslayer(nothing, ps, st)
        reduce(vcat, map(zip(sols[1:end-1], keys(ax[1])[2:end])) do (sol, name_i)
            _u0 = getproperty(ps, name_i).u0
            sol.states[:,end] .-_u0
        end)
    end
end

ll = shooting_constraints(ps, nothing)

eq_cons(res, x, p) = res .= shooting_constraints(x, p)

jac_fd = ForwardDiff.jacobian(Base.Fix2(shooting_constraints, nothing), collect(ps))
spy(jac_fd)

lb, ub = copy(ps), copy(ps)
ub.controls .= 1.0
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
    loss, AutoForwardDiff(), #cons = eq_cons
)

optprob = OptimizationProblem(
    optfun, collect(ps), lb = collect(lb), ub = collect(ub)#, lcons = zero(ll), ucons=zero(ll)
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

popt = uopt + zero(ps)

uopt = popt.controls[1:48]
w1opt = popt.controls[49:48+48]
w2opt = popt.controls[97:end]

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