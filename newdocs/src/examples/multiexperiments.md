# Multiexperiment example

This page shows how to design multiple experiments at once using the `MultiExperimentLayer` again at the example of the Lotka-Volterra system. For a general introduction, look again at [Optimal experimental design example](@ref).

```@example lotka_multi
using Corleone
using Corleone.LuxCore, Corleone.ComponentArrays
using OrdinaryDiffEqTsit5
using SciMLSensitivity
using Random, CairoMakie
using Optimization, OptimizationMOI, Ipopt

function lotka(u, p, t)
    return [u[1] - p[1] * prod(u[1:2]) - 0.4 * p[3] * u[1];
            -u[2] + p[2] * prod(u[1:2]) - 0.2 * p[3] * u[2]]
end

tspan = (0., 12.)
u0 = [0.5, 0.7]
p = [1.0, 1.0, 0.0]

prob = ODEProblem(lotka, u0, tspan, p)
control_grid = collect(0.0:0.1:11.9)
control = ControlParameter(
    control_grid, name = :fishing, bounds=(0.0,1.0)
)
```

As in [Optimal experimental design example](@ref), we first define an `OEDLayer`. However, in contrast to before, now we also leave the initial conditions `u0``as degrees of freedom to be optimized.

```@example lotka_multi
oed_layer = Corleone.OEDLayer(prob, Tsit5(); params=[1,2], controls = (control,),
            tunable_ic = [1,2], bounds_ic = ([0.3, 0.3], [0.9,0.9]),
            control_indices = [3], dt = 0.25)
```

With this `OEDLayer` we construct a `MultiExperimentLayer` with 2 experiments that we want to design.

```@example lotka_multi
nexp = 2
multi_exp = MultiExperimentLayer(oed_layer, nexp)

ps, st = LuxCore.setup(Random.default_rng(), multi_exp)
p = ComponentArray(ps)
lb, ub = Corleone.get_bounds(multi_exp)

multi_exp(nothing, p, st)
```

To set up the `OptimizationProblem`, we need to define the objective function and the sampling constraints, i.e., the upper bound on the measurements. Again, the objective can be easily set up by calling one of the provided criteria on the `MultiExperimentLayer`.

```@example lotka_multi
crit = ACriterion()
criterion = crit(multi_exp)
criterion(p, nothing)
```

The constraints can be directly stated as linear constraints on the discretized sampling controls. For this we need to sum up the discretized controls of each measurement function. This time, however, we need to do this for all experiments.

```@example lotka_multi
nc = vcat(0, cumsum([length(x.t) for x in oed_layer.layer.controls]))

sampling_cons = let ax=getaxes(p), dt=0.25
    (res, p, ::Any) -> begin
        ps = ComponentArray(p, ax)
        res .= [
            sum(ps.experiment_1.controls[nc[2]+1:nc[3]]) * dt;
            sum(ps.experiment_1.controls[nc[3]+1:nc[4]]) * dt;
            sum(ps.experiment_2.controls[nc[2]+1:nc[3]]) * dt;
            sum(ps.experiment_2.controls[nc[3]+1:nc[4]]) * dt
          ]
    end
end

sampling_cons(zeros(4), p, nothing)
```


All sampling controls are initialized with ``w=1``, therefore the constraint is evaluated as the length of the complete time horizon of the problem. We now specify that we may measure four time units and solve the problem.

```@example lotka_multi

optfun = OptimizationFunction(
    criterion, AutoForwardDiff(), cons = sampling_cons
)

optprob = OptimizationProblem(
    optfun, collect(p), lb = collect(lb), ub = collect(ub), lcons=zeros(2*nexp), ucons=4.0*ones(2*nexp)
)

uopt = solve(optprob, Ipopt.Optimizer(),
     tol = 1e-9,
     hessian_approximation = "limited-memory",
     max_iter = 300
)
```

The problem is efficiently solved. Lastly, we have a look at the solution.

```@example lotka_multi
optsol, _ = multi_exp(nothing, uopt + zero(p), st)

f = Figure(size = (800,800))
for i = 1:nexp
    ax = CairoMakie.Axis(f[1,i], xticks=0:2:12, title="Experiment $i")
    ax1 = CairoMakie.Axis(f[2,i], xticks=0:2:12)
    ax2 = CairoMakie.Axis(f[3,i], xticks=0:2:12)
    ax3 = CairoMakie.Axis(f[4,i], xticks=0:2:12, limits=(nothing, (-0.05,1.05)))
    [plot!(ax, optsol[i].t, sol) for sol in eachrow(Array(optsol[i]))[1:2]]
    [plot!(ax1, optsol[i].t, sol) for sol in eachrow(reduce(hcat, (optsol[i][Corleone.sensitivity_variables(multi_exp.layers)[:]])))]
    [plot!(ax2, optsol[i].t, sol) for sol in eachrow(reduce(hcat, (optsol[i][Corleone.fisher_variables(multi_exp.layers)])))]
    stairs!(ax, control.t,  getproperty(uopt + zero(p), Symbol("experiment_$i")).controls[nc[1]+1:nc[2]], color=:black)
    stairs!(ax3, 0.0:0.25:11.75, getproperty(uopt + zero(p), Symbol("experiment_$i")).controls[nc[2]+1:nc[3]])
    stairs!(ax3, 0.0:0.25:11.75, getproperty(uopt + zero(p), Symbol("experiment_$i")).controls[nc[3]+1:nc[4]])
end
CairoMakie.save("lotka_multi.svg",f) # hide
```

![](lotka_multi.svg)