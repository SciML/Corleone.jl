using Pkg
Pkg.activate(joinpath(@__DIR__, "../"))


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
using LinearAlgebra

function lotka_dynamics(u, p, t)
    return [u[1] - p[2] * prod(u[1:2]) - 0.4 * p[1] * u[1];
            -u[2] + p[3] * prod(u[1:2]) - 0.2 * p[1] * u[2]]
end

tspan = (0., 12.)
u0 = [0.5, 0.7, ]
p0 = [0.0, 1.0, 1.0]
prob = ODEProblem{false}(lotka_dynamics, u0, tspan, p0,
    sensealg = ForwardDiffSensitivity()
    )
control = ControlParameter(
    collect(0.0:0.25:12.0)[1:end-1], name = :fishing
)

# Single Shooting
layer = CorleoneCore.SingleShootingLayer(prob, Tsit5(),Int64[],[1], (control,))
oed_layer = CorleoneCore.augment_layer_for_oed(layer; observed = (u,p,t) -> u[1:2])
ps, st = LuxCore.setup(Random.default_rng(), oed_layer)
p = ComponentArray(ps)

nc, dt = length(control.t), diff(control.t)[1]

loss = let layer = oed_layer, st = st, ax = getaxes(p)
    (p, ::Any) -> begin
        ps = ComponentArray(p, ax)
        sols, _ = layer(nothing, ps, st)
        inv(tr(reshape(sols.states[end-1:end,end-1:end], (2,2))))
    end
end

loss(collect(p), nothing)

sampling_cons = let layer = oed_layer, st = st, ax = getaxes(p)
    (res, p, ::Any) -> begin
        ps = ComponentArray(p, ax)
        res .= [sum(ps.controls[nc+1:2*nc]) * dt;
          sum(ps.controls[2*nc+1:3*nc]) * dt  ]
    end
end

sampling_cons(zeros(2),collect(p), nothing)

optfun = OptimizationFunction(
    loss, AutoForwardDiff(), cons = sampling_cons
)

lb, ub = copy(p), copy(p)
lb.controls .= 0.0
ub.controls .= 1.0

optprob = OptimizationProblem(
    optfun, collect(p), lb = collect(lb), ub = collect(ub), lcons=zeros(2), ucons=[4.0, 4.0]
)

uopt = solve(optprob, Ipopt.Optimizer(),
     tol = 1e-6,
     hessian_approximation = "limited-memory",
     max_iter = 300
)

optsol, _ = oed_layer(nothing, uopt + zero(p), st)

f = Figure()
ax = CairoMakie.Axis(f[1,1])
ax1 = CairoMakie.Axis(f[2,1])
ax2 = CairoMakie.Axis(f[1,2])
ax3 = CairoMakie.Axis(f[2,2])
[plot!(ax, optsol.time, sol) for sol in eachrow(optsol.states)[1:2]]
[plot!(ax1, optsol.time, sol) for sol in eachrow(optsol.states)[3:8]]
[plot!(ax2, optsol.time, sol) for sol in eachrow(optsol.states)[9:end]]
stairs!(ax3, control.t, (uopt + zero(p)).controls[1:length(control.t)])
stairs!(ax3, control.t, (uopt + zero(p)).controls[length(control.t)+1:2*length(control.t)])
stairs!(ax3, control.t, (uopt + zero(p)).controls[2*length(control.t)+1:3*length(control.t)])
f


## Multiple Shooting
shooting_points = [0.0,4.0, 8.0, 12.0]
mslayer = CorleoneCore.MultipleShootingLayer(prob, Tsit5(),Int64[],[1], (control,), shooting_points)
oed_mslayer = CorleoneCore.augment_layer_for_oed(mslayer; observed = (u,p,t) -> u[1:2])

msps, msst = LuxCore.setup(Random.default_rng(), oed_mslayer)
# Or use any of the Initialization schemes
msps, msst = ForwardSolveInitialization()(Random.default_rng(), oed_mslayer)
msp = ComponentArray(msps)

msloss = let layer = oed_mslayer, st = msst, ax = getaxes(msp)
    (p, ::Any) -> begin
        ps = ComponentArray(p, ax)
        sols, _ = layer(nothing, ps, st)
        inv(tr(reshape(last(sols).states[end-1:end,end-1:end], (2,2))))
    end
end

nc_ms = length(first(oed_mslayer.layers).controls[1].t)
shooting_constraints = let layer = oed_mslayer, st = msst, ax = getaxes(msp)
    (p, ::Any) -> begin
        ps = ComponentArray(p, ax)
        sols, _ = layer(nothing, ps, st)
        matching_ = reduce(vcat, map(zip(sols[1:end-1], keys(ax[1])[2:end])) do (sol, name_i)
            _u0 = getproperty(ps, name_i).u0
            sol.states[:,end] .-_u0
        end)
        sampling_ = [sum(reduce(vcat, [ps["layer_$i"].controls[nc_ms+1:2*nc_ms] for i in 1:length(layer.layers)])) * dt;
                    sum(reduce(vcat, [ps["layer_$i"].controls[2*nc_ms+1:3*nc_ms] for i in 1:length(layer.layers)])) *  dt]
        return vcat(matching_, sampling_)
    end
end

matching = shooting_constraints(msp, nothing)
jac_cons = ForwardDiff.jacobian(Base.Fix2(shooting_constraints, nothing), msp)
eq_cons(res, x, p) = res .= shooting_constraints(x, p)

ms_lb = map(msps) do p_i
    p_i.controls .= 0.0
    p_i.u0 .= -500.0
    p_i
end |> ComponentArray


ms_ub = map(msps) do p_i
    p_i.controls .= 1.0
    p_i.u0 .= 1000.0
    p_i
end |> ComponentArray


optfun = OptimizationFunction(
    msloss, AutoForwardDiff(), cons = eq_cons
)

ucons = zero(matching)
ucons[end-1:end] .= 4.0
optprob = OptimizationProblem(
    optfun, collect(msp), lb = collect(ms_lb), ub = collect(ms_ub), lcons = zero(matching), ucons=ucons
)

uopt = solve(optprob, Ipopt.Optimizer(),
     tol = 1e-6,
     hessian_approximation = "limited-memory",
     max_iter = 300 # 165
)

blocks = CorleoneCore.get_block_structure(oed_mslayer)

uopt = solve(optprob, BlockSQPOpt(),
    opttol = 1e-6,
    options = blockSQP.sparse_options(),
    sparsity = blocks,
    maxiters = 300 # 165
)


sol_u = uopt + zero(msp)
lc = first(oed_mslayer.layers).controls[1].t |> length

fishing_opt, w1_opt, w2_opt = map(1:length(oed_mslayer.layers)) do layer
    u_local = sol_u["layer_$layer"].controls
    u_local[1:lc], u_local[lc+1:2*lc], u_local[2*lc+1:3*lc]
end

mssol, _ = oed_mslayer(nothing, uopt + zero(msp), msst)

f = Figure()
ax = CairoMakie.Axis(f[1,1])
ax1 = CairoMakie.Axis(f[2,1])
[plot!(ax, sol.time, sol.states[i,:], color=i)  for sol in mssol for i =1:size(sol.states,1)]

[stairs!(ax1, c.controls[1].t, (uopt + zero(msp))["layer_$i"].controls[1:lc], color=:black) for (i,c) in enumerate(mslayer.layers)]
[stairs!(ax1, c.controls[1].t, (uopt + zero(msp))["layer_$i"].controls[lc+1:2*lc], color=:red) for (i,c) in enumerate(mslayer.layers)]
[stairs!(ax1, c.controls[1].t, (uopt + zero(msp))["layer_$i"].controls[2*lc+1:3*lc], color=:yellow) for (i,c) in enumerate(mslayer.layers)]

f