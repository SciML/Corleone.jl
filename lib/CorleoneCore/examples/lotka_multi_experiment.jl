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
    collect(0.0:0.25:11.75), name = :fishing, bounds = (0.,1.)
)



# MultiExperiment with ALL FIXED, JUST SAMPLING TIMES
ol = OEDLayer(prob, Tsit5(); params= [2,3], dt = 0.2)

multi_exp = MultiExperimentLayer(ol, 2)



ps, st = LuxCore.setup(Random.default_rng(), multi_exp)
pps = ComponentArray(ps)
lb, ub = CorleoneCore.get_bounds(multi_exp)

single_sol, _ = ol(nothing, ps.experiment_1, st.experiment_1)
single_sol.u

sols, _ = multi_exp(nothing, ps, st)

crit= ACriterion()
ACrit = crit(multi_exp)

ACrit(pps, nothing)

nc = vcat(0, cumsum([length(c.t) for c in ol.layer.controls])...)

sampling_cons = let ax = getaxes(pps), nc = nc, dt = 0.2
    (res, p, ::Any) -> begin
        ps = ComponentArray(p, ax)
        wexp1 = [sum(ps.experiment_1.controls[nc[i]+1:nc[i+1]]) * dt for i in eachindex(nc)[1:end-1]]
        wexp2 = [sum(ps.experiment_2.controls[nc[i]+1:nc[i+1]]) * dt for i in eachindex(nc)[1:end-1]]
        res .= vcat(wexp1, wexp2)
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

f = Figure()
ax = CairoMakie.Axis(f[1,1], xticks = 0:2:12)
ax2 = CairoMakie.Axis(f[1,2], xticks = 0:2:12)
ax3 = CairoMakie.Axis(f[2,1], xticks = 0:2:12, title="Experiment 1")
ax4 = CairoMakie.Axis(f[2,2], xticks = 0:2:12, title="Experiment 2")
[plot!(ax, optsol[1].t, sol) for sol in eachrow(Array(optsol[1]))[1:2]]
[plot!(ax2, optsol[1].t, sol) for sol in eachrow(reduce(hcat, (optsol[1][CorleoneCore.sensitivity_variables(multi_exp.layer)])))]
[stairs!(ax3, last(ol.layer.controls).t, sampling_opt.experiment_1.controls[nc[i]+1:nc[i+1]]) for i=1:2]
[stairs!(ax4, last(ol.layer.controls).t, sampling_opt.experiment_2.controls[nc[i]+1:nc[i+1]]) for i=1:2]
f


# Single Shooting
oed_layer = CorleoneCore.OEDLayer(prob, Tsit5(); params=[2,3], controls = (control,),
            tunable_ic = [1,2], bounds_ic = ([0.3, 0.3], [0.9,0.9]),
            control_indices = [1], dt = 0.25)

nexp = 2
multi_exp = MultiExperimentLayer(oed_layer, nexp)

ps, st = LuxCore.setup(Random.default_rng(), multi_exp)
p = ComponentArray(ps)
lb, ub = CorleoneCore.get_bounds(multi_exp)
nc, dt = length(control.t), diff(control.t)[1]

sols, _ = multi_exp(nothing, ps, st)

criterion = DCriterion()(multi_exp)
criterion(p, nothing)

ForwardDiff.gradient(Base.Fix2(criterion, nothing), p)

sampling_cons = let ax = getaxes(p)
    (res, p, ::Any) -> begin
        ps = ComponentArray(p, ax)
        res .= [
            sum(ps.experiment_1.controls[nc+1:2*nc]) * dt;
            sum(ps.experiment_2.controls[nc+1:2*nc]) * dt;
            #sum(ps.experiment_3.controls[nc+1:2*nc]) * dt;
           # sum(ps.experiment_4.controls[nc+1:2*nc]) * dt;
            sum(ps.experiment_1.controls[2*nc+1:3*nc]) * dt  ;
            sum(ps.experiment_2.controls[2*nc+1:3*nc]) * dt  ;
            #sum(ps.experiment_3.controls[2*nc+1:3*nc]) * dt  ;
            #sum(ps.experiment_4.controls[2*nc+1:3*nc]) * dt  ;
          ]
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
     #tol = 1e-10,
     hessian_approximation = "limited-memory",
     max_iter = 300
)

optu = uopt + zero(p)

optsol, _ = multi_exp(nothing, optu, st)

f = Figure()
for i = 1:nexp
    ax = CairoMakie.Axis(f[1,i], xticks=0:2:12, title="Experiment $i")
    ax1 = CairoMakie.Axis(f[2,i], xticks=0:2:12)
    ax2 = CairoMakie.Axis(f[3,i], xticks=0:2:12)
    ax3 = CairoMakie.Axis(f[4,i], xticks=0:2:12, limits=(nothing, (-0.05,1.05)))
    [plot!(ax, optsol[i].t, sol) for sol in eachrow(Array(optsol[i]))[1:2]]
    [plot!(ax1, optsol[i].t, sol) for sol in eachrow(reduce(hcat, (optsol[i][CorleoneCore.sensitivity_variables(multi_exp.layer)])))]
    [plot!(ax2, optsol[i].t, sol) for sol in eachrow(reduce(hcat, (optsol[i][CorleoneCore.fisher_variables(multi_exp.layer)])))]
    stairs!(ax, control.t,  getproperty(uopt + zero(p), Symbol("experiment_$i")).controls[1:length(control.t)], color=:black)
    stairs!(ax3, control.t, getproperty(uopt + zero(p), Symbol("experiment_$i")).controls[length(control.t)+1:2*length(control.t)])
    stairs!(ax3, control.t, getproperty(uopt + zero(p), Symbol("experiment_$i")).controls[2*length(control.t)+1:3*length(control.t)])
end
f


## Multiple Shooting
shooting_points = [0.0,4.0, 8.0, 12.0]
oed_mslayer = OEDLayer(prob, Tsit5(), shooting_points; params=[2,3], dt = 0.25,
            tunable_ic = [1,2], bounds_ic = ([0.3, 0.3], [0.9,0.9]),
            control_indices = [1], controls=(control,),
            bounds_nodes = (0.05 * ones(2), 10*ones(2)))

multi_exp = MultiExperimentLayer(oed_mslayer, nexp)


oed_msps, oed_msst = LuxCore.setup(Random.default_rng(), oed_mslayer)
oed_msps, oed_msst = LuxCore.setup(Random.default_rng(), multi_exp)
# Or use any of the provided Initialization schemes
#oed_msps, oed_msst = ForwardSolveInitialization()(Random.default_rng(), oed_mslayer)
oed_msp = ComponentArray(oed_msps)
oed_ms_lb, oed_ms_ub = CorleoneCore.get_bounds(multi_exp)
oed_sols, _ = multi_exp(nothing, oed_msp, oed_msst)

crit = DCriterion()
criterion = crit(multi_exp)
criterion(oed_msp, nothing)

matching_multi = CorleoneCore.get_shooting_constraints(multi_exp)

matching_multi(oed_sols, oed_msp)


function extract_and_join_controls(layer::OEDLayer, p)
    ps, st = LuxCore.setup(Random.default_rng(), layer)

    _p = isa(p, Array) ? ComponentArray(p, getaxes(ComponentArray(ps))) : p
    controls, control_idxs = CorleoneCore.get_controls(layer.layer)
    starts_controls = zeros(Int, length(controls))

    joined_controls = [[] for i=1:length(controls)]
    for (layer_i, sublayer) in enumerate(layer.layer.layers)
        layer_controls, _ = CorleoneCore.get_controls(sublayer)
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

function extract_and_join_controls(layer::CorleoneCore.MultiExperimentLayer, p)
    ps, st = LuxCore.setup(Random.default_rng(), layer)
    _p = isa(p, Array) ? ComponentArray(_p, getaxes(ComponentArray(ps))) : p
    joined_controls = map(1:layer.n_exp) do i
        local_ps = getproperty(_p, Symbol("experiment_$i"))
        extract_and_join_controls(layer.layer, local_ps)
    end
    exp_names = Tuple([Symbol("experiment_$i") for i=1:layer.n_exp])
    NamedTuple{exp_names}(Tuple(joined_controls))
end

extract_and_join_controls(multi_exp, oed_msps)



shooting_constraints = let layer = multi_exp, dt = 0.25, st = oed_msst, ax = getaxes(oed_msp), matching_constraint = CorleoneCore.get_shooting_constraints(multi_exp)
    (p, ::Any) -> begin
        ps = ComponentArray(p, ax)
        sols, _ = layer(nothing, ps, st)
        matching_ = matching_constraint(sols, ps)
        joined_controls = extract_and_join_controls(layer, ps)
        sampling_ = [
            sum( joined_controls.experiment_1[2] * dt);
            sum( joined_controls.experiment_1[3] * dt)
            sum( joined_controls.experiment_2[2] * dt)
            sum( joined_controls.experiment_2[3] * dt)
            ]
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
    optfun, collect(oed_msp), lb = collect(oed_ms_lb), ub = collect(oed_ms_ub), lcons = zero(constraints_eval), ucons=ucons
)

uopt = solve(optprob, Ipopt.Optimizer(),
     tol = 1e-6,
     hessian_approximation = "limited-memory",
     max_iter = 300 # 165
)

#blocks = CorleoneCore.get_block_structure(oed_mslayer)

#uopt = solve(optprob, BlockSQPOpt(),
#    opttol = 1e-6,
#    options = blockSQP.sparse_options(),
#    sparsity = blocks,
#    maxiters = 300 # 165
#)


sol_u = uopt + zero(oed_msp)



mssol, _ = multi_exp(nothing, sol_u, oed_msst)

joint_controls = extract_and_join_controls(multi_exp, sol_u)
ct = 0:0.25:11.75 |> collect
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