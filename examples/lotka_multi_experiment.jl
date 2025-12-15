using Pkg
Pkg.activate(@__DIR__)
using Corleone
using OrdinaryDiffEq
using SciMLSensitivity
using ComponentArrays
using LuxCore
using Random

using CairoMakie
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
    collect(0.0:0.25:11.75), name = :fishing, bounds = (0.,1.)
)

# MultiExperiment with ALL FIXED, JUST SAMPLING TIMES
ol = OEDLayer(prob, Tsit5(); params= [2,3], dt = 0.2)

multi_exp = MultiExperimentLayer(ol, 2)

ps, st = LuxCore.setup(Random.default_rng(), multi_exp)
pps = ComponentArray(ps)
lb, ub = Corleone.get_bounds(multi_exp)

sampling = get_sampling_constraint(multi_exp)

sols, _ = multi_exp(nothing, ps, st)

crit= ACriterion()
ACrit = crit(multi_exp)
ACrit(pps, nothing)

sampling_cons = let ax = getaxes(pps), sampling=sampling
    (res, p, ::Any) -> begin
        ps = ComponentArray(p, ax)
        res .= sampling(ps, nothing)
    end
end

sampling_cons(zeros(4), pps, st)

optfun = OptimizationFunction(
   ACrit, AutoReverseDiff(), cons = sampling_cons
)

optprob = OptimizationProblem(
    optfun, collect(pps), lb = collect(lb), ub = collect(ub), lcons=zeros(4), ucons=zeros(4) .+ 4.0
)

uopt = solve(optprob, Ipopt.Optimizer(),
    hessian_approximation = "limited-memory",
    max_iter = 300
)

sampling_opt = uopt + zero(pps)
optsol, _ = multi_exp(nothing, sampling_opt, st)

nc = Corleone.control_blocks(multi_exp)
f = Figure()
ax = CairoMakie.Axis(f[1,1], xticks = 0:2:12)
ax2 = CairoMakie.Axis(f[1,2], xticks = 0:2:12)
ax3 = CairoMakie.Axis(f[2,1], xticks = 0:2:12, title="Experiment 1")
ax4 = CairoMakie.Axis(f[2,2], xticks = 0:2:12, title="Experiment 2")
[plot!(ax, optsol[1].t, sol) for sol in eachrow(Array(optsol[1]))[1:2]]
[plot!(ax2, optsol[1].t, sol) for sol in eachrow(reduce(hcat, (optsol[1][Corleone.sensitivity_variables(multi_exp.layers)[:]])))]
[stairs!(ax3, last(ol.layer.controls).t, sampling_opt.experiment_1.controls[nc[i]+1:nc[i+1]]) for i=1:2]
[stairs!(ax4, last(ol.layer.controls).t, sampling_opt.experiment_2.controls[nc[i]+1:nc[i+1]]) for i=1:2]
f


# Single Shooting
dt = 0.25
oed_layer = Corleone.OEDLayer(prob, Tsit5(); params=[2,3], controls = (control,),
            tunable_ic = [1,2], bounds_ic = ([0.3, 0.3], [0.9,0.9]),
            control_indices = [1], dt = dt)

nexp = 2
multi_exp = MultiExperimentLayer(oed_layer, nexp)

ps, st = LuxCore.setup(Random.default_rng(), multi_exp)
p = ComponentArray(ps)
lb, ub = Corleone.get_bounds(multi_exp)

sols, _ = multi_exp(nothing, ps, st)

criterion = ACriterion()(multi_exp)
criterion(p, nothing)

sampling = get_sampling_constraint(multi_exp)

sampling_cons = let ax = getaxes(p), sampling=sampling
    (res, p, ::Any) -> begin
        ps = ComponentArray(p, ax)
        res .= sampling(ps, nothing)
    end
end

sampling_cons(zeros(nexp*2),collect(p), nothing)

optfun = OptimizationFunction(
    criterion, AutoForwardDiff(), cons = sampling_cons
)

optprob = OptimizationProblem(
    optfun, collect(p), lb = collect(lb), ub = collect(ub), lcons=zeros(nexp * 2), ucons=zeros(nexp*2) .+ 4.0
)

uopt = solve(optprob, Ipopt.Optimizer(),
     tol = 1e-7,
     hessian_approximation = "limited-memory",
     max_iter = 300
)

block_structure = Corleone.get_block_structure(multi_exp)

uopt = solve(optprob, BlockSQPOpt(),
    opttol = 1e-8,
    options = blockSQP.sparse_options(),
    sparsity = block_structure,
    maxiters = 300 # 165
)

optu = uopt + zero(p)

optsol, _ = multi_exp(nothing, optu, st)

nc = Corleone.control_blocks(multi_exp)
f = Figure(size = (800,800))
for i = 1:nexp
    ax = CairoMakie.Axis(f[1,i], xticks=0:2:12, title="Experiment $i")
    ax1 = CairoMakie.Axis(f[2,i], xticks=0:2:12)
    ax2 = CairoMakie.Axis(f[3,i], xticks=0:2:12)
    ax3 = CairoMakie.Axis(f[4,i], xticks=0:2:12, limits=(nothing, (-0.05,1.05)))
    [plot!(ax, optsol[i].t, sol) for sol in eachrow(Array(optsol[i]))[1:2]]
    [plot!(ax1, optsol[i].t, sol) for sol in eachrow(reduce(hcat, (optsol[i][Corleone.sensitivity_variables(multi_exp.layers)[:]])))]
    [plot!(ax2, optsol[i].t, sol) for sol in eachrow(reduce(hcat, (optsol[i][Corleone.fisher_variables(multi_exp.layers)])))]
    stairs!(ax, control.t,  getproperty(uopt + zero(p), Symbol("experiment_$i")).controls[1:length(control.t)], color=:black)
    stairs!(ax3, 0.0:dt:12.0-dt, getproperty(uopt + zero(p), Symbol("experiment_$i")).controls[nc[2]+1:nc[3]])
    stairs!(ax3, 0.0:dt:12.0-dt, getproperty(uopt + zero(p), Symbol("experiment_$i")).controls[nc[3]+1:nc[4]])
end
display(f)

IG = Corleone.InformationGain(multi_exp, uopt.u)
multiplier = uopt.original.inner.mult_g #multiplier
multiplier = uopt.original.multiplier

f_IG = Figure(size = (800,800))
for i = 1:nexp
    local_IG = getproperty(IG, Symbol("experiment_$i"))
    ax = CairoMakie.Axis(f_IG[1,i], xticks=0:2:12, title="Experiment $i")
    ax1 = CairoMakie.Axis(f_IG[2,i], xticks=0:2:12)
    lines!(ax, local_IG.t, tr.(local_IG.global_information_gain[1]))
    hlines!(ax, multiplier[(i-1)*2+1], color=:black)
    lines!(ax1, local_IG.t, tr.(local_IG.global_information_gain[2]))
    hlines!(ax1, multiplier[(i-1)*2+2], color=:black)
end
display(f_IG)

## Multiple Shooting
shooting_points = [0.0,4.0, 8.0, 12.0]
oed_mslayer = OEDLayer(prob, Tsit5(), shooting_points; params=[2,3], dt = dt,
            tunable_ic = [1,2], bounds_ic = ([0.3, 0.3], [0.9,0.9]),
            control_indices = [1], controls=(control,),
            bounds_nodes = (0.05 * ones(2), 10*ones(2)))

multi_exp = MultiExperimentLayer(oed_mslayer, nexp)


oed_msps, oed_msst = LuxCore.setup(Random.default_rng(), oed_mslayer)
oed_msps, oed_msst = LuxCore.setup(Random.default_rng(), multi_exp)
# Or use any of the provided Initialization schemes
oed_msps, oed_msst = ForwardSolveInitialization()(Random.default_rng(), multi_exp)
oed_msp = ComponentArray(oed_msps)
oed_ms_lb, oed_ms_ub = Corleone.get_bounds(multi_exp)
oed_sols, _ = multi_exp(nothing, oed_msp, oed_msst)

crit = ACriterion()
criterion = crit(multi_exp)
criterion(oed_msp, nothing)

matching_multi = Corleone.get_shooting_constraints(multi_exp)
matching_multi(oed_sols, oed_msp)

sampling = get_sampling_constraint(multi_exp)
sampling(oed_msp, oed_msst)

function extract_and_join_controls(layer::OEDLayer, p)
    ps, st = LuxCore.setup(Random.default_rng(), layer)

    _p = isa(p, Array) ? ComponentArray(p, getaxes(ComponentArray(ps))) : p
    controls, control_idxs = Corleone.get_controls(layer.layer)
    starts_controls = zeros(Int, length(controls))

    joined_controls = [[] for i=1:length(controls)]
    for (layer_i, sublayer) in enumerate(layer.layer.layers)
        layer_controls, _ = Corleone.get_controls(sublayer)
        local_tspan = sublayer.problem.tspan
        local_ps = getproperty(_p, Symbol("layer_$layer_i"))
        for (i, ci) in enumerate(layer_controls)
            idxs = findall(x -> first(local_tspan) <= x < last(local_tspan), ci.t)
            if length(idxs) == 0
                continue
            end

            append!(joined_controls[i], local_ps.controls[starts_controls[layer_i]+1:starts_controls[layer_i]+length(idxs)])
            starts_controls[layer_i] += length(idxs)
        end
    end
    return joined_controls
end

function extract_and_join_controls(layer::Corleone.MultiExperimentLayer, p)
    ps, st = LuxCore.setup(Random.default_rng(), layer)
    _p = isa(p, Array) ? ComponentArray(_p, getaxes(ComponentArray(ps))) : p
    joined_controls = map(1:layer.n_exp) do i
        local_ps = getproperty(_p, Symbol("experiment_$i"))
        extract_and_join_controls(layer.layers, local_ps)
    end
    exp_names = Tuple([Symbol("experiment_$i") for i=1:layer.n_exp])
    NamedTuple{exp_names}(Tuple(joined_controls))
end

extract_and_join_controls(multi_exp, oed_msps)



shooting_constraints = let layer = multi_exp, st = oed_msst, ax = getaxes(oed_msp), sampling=sampling, matching= matching_multi
    (p, ::Any) -> begin
        ps = ComponentArray(p, ax)
        sols, _ = layer(nothing, ps, st)
        matching_ = matching(sols, ps)
        sampling_ = sampling(ps, st)
        return vcat(matching_, sampling_)
    end
end

eq_cons(res, x, p) = res .= shooting_constraints(x, p)
shooting_constraints(oed_msp, nothing)

optfun = OptimizationFunction(
    criterion, AutoForwardDiff(), cons = eq_cons
)

constraints_eval = shooting_constraints(oed_msp, nothing)
ucons = zero(constraints_eval)
ucons[end-3:end] .= 4.0
optprob = OptimizationProblem(
    optfun, collect(oed_msp), lb = collect(oed_ms_lb), ub = collect(oed_ms_ub),
            lcons = zero(constraints_eval), ucons=ucons
)

uopt = solve(optprob, Ipopt.Optimizer(),
     tol = 1e-6,
     hessian_approximation = "limited-memory",
     max_iter = 300 # 165
)

block_structure_ms = Corleone.get_block_structure(multi_exp)

uopt = solve(optprob, BlockSQPOpt(),
    opttol = 1e-6,
    options = blockSQP.sparse_options(),
    sparsity = block_structure_ms,
    maxiters = 300 # 165
)


sol_u = uopt + zero(oed_msp)

mssol, _ = multi_exp(nothing, sol_u, oed_msst)

joint_controls = extract_and_join_controls(multi_exp, sol_u)
ct = 0:dt:12.0 - dt |> collect
f = Figure()
for i=1:nexp

    ax = CairoMakie.Axis(f[1,i], limits=(nothing, (0, 5)))
    ax1 = CairoMakie.Axis(f[2,i], limits = (nothing, (-15, 15)))
    ax2 = CairoMakie.Axis(f[3,i], limits = (nothing, (-10, 250)))
    ax3 = CairoMakie.Axis(f[4,i])
    [plot!(ax,  sol.t, Array(sol)[i,:])  for sol in mssol[i] for i in 1:2]
    [plot!(ax1, sol.t, Array(sol)[i,:])  for sol in mssol[i] for i in 3:6]
    [plot!(ax2, sol.t, Array(sol)[i,:])  for sol in mssol[i] for i in 7:9]
    f

    stairs!(ax,  ct, joint_controls[i][1], color=:black)
    stairs!(ax3, ct, joint_controls[i][2], color=Makie.wong_colors()[1])
    stairs!(ax3, ct, joint_controls[i][3], color=Makie.wong_colors()[2])

end
f


IG = Corleone.InformationGain(multi_exp, uopt.u)
multiplier = uopt.original.inner.mult_g[end-3:end] #multiplier
multiplier = uopt.original.multiplier

f_IG = Figure(size = (800,800))
for i = 1:nexp
    local_IG = getproperty(IG, Symbol("experiment_$i"))
    ax = CairoMakie.Axis(f_IG[1,i], xticks=0:2:12, title="Experiment $i")
    ax1 = CairoMakie.Axis(f_IG[2,i], xticks=0:2:12)
    lines!(ax, local_IG.t, tr.(local_IG.global_information_gain[1]))
    hlines!(ax, multiplier[(i-1)*2+1], color=:black)
    lines!(ax1, local_IG.t, tr.(local_IG.global_information_gain[2]))
    hlines!(ax1, multiplier[(i-1)*2+2], color=:black)
end
display(f_IG)