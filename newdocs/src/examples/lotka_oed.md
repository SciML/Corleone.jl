# Optimal experimental design example

This page shows how to formulate optimal experimental design problems in Corleone. The problem is built around the Lotka-Volterra system we have already seen in [Simple optimal control example](@ref).
However, it is now the goal to estimate the parameters ``p_1`` and ``p_2`` accurately from data. For this, experiments need to be designed which yield the most information on these parameters. The problem that achieves this reads

```math
 \begin{array}{lll}
 \displaystyle \min_{x,G,F,u,w^1,w^2} && \text{trace} \; \left( F^{-1}(t_f) \right) \\
 \text{subject to} \\
\quad \dot{x_1}(t) & = &  x_1(t) - p_1 x_1(t) x_2(t) - c_1 u(t) x_1(t),\\
\quad \dot{x_2}(t) & = &  - x_2(t) + p_2 x_1(t) x_2(t) - c_2 u(t) x_2(t),\\
\quad \dot{G_{11}}(t) & = & f_{x11}(\cdot) \; G_{11}(t) + f_{x12}(\cdot) \; G_{21}(t) + f_{p12}(\cdot), \\
\quad \dot{G_{12}}(t) & = & f_{x11}(\cdot) \; G_{12}(t) + f_{x12}(\cdot) \; G_{22}(t), \\
\quad \dot{G_{21}}(t) & = & f_{x21}(\cdot) \; G_{11}(t) + f_{x22}(\cdot) \; G_{21}(t), \\
\quad \dot{G_{22}}(t) & = & f_{x21}(\cdot) \; G_{12}(t) + f_{x22}(\cdot) \; G_{22}(t) + f_{p24}(\cdot), \\
\quad \dot{F_{11}}(t) & = & w_1(t) G_{11}(t)^2 + w_2(t) G_{21}(t)^2, \\
\quad \dot{F_{12}}(t) & = & w_1(t) G_{11}(t) G_{12}(t) + w_2(t) G_{21}(t) G_{22}(t), \\
\quad \dot{F_{22}}(t) & = & w_1(t) G_{12}(t)^2 + w_2(t) G_{22}(t)^2, \\
\quad \dot{z_1}(t) & = & w_1(t), \\
\quad \dot{z_2}(t) & = & w_2(t), \\[1.5ex]
\quad x(0) &=& (0.5, 0.7), \\
\quad G(0) &=& F(0) = 0, \\
\quad z(0) 0, \\[1.5ex]
\quad u(t) & \in & \mathcal{U}, \; w_1(t), w_2(t) \in \mathcal{W} \\
\quad 0    & \le & M - z(t_f)
  \end{array}
```

The dynamical system of the two species  is augmented with the sensitivities ``G`` of the parameters ``p_1`` and ``p_2`` and the Fisher information matrix ``F``. The objective function is the ACriterion, i.e., the trace of the inverse of the Fisher information matrix. Moreover, we assume that we can measure the two states directly and have an upper bound of ``M`` on the measurements.

The first step to model and solve this problem in Corleone is to define the underlying differential equation problem and the control discretization of ``u``.

```@example lotka_oed
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

Now, we need to augment the dynamical system with the sensitivities and Fisher information matrix corresponding to the two parameters of interest. We do this via building an `OEDLayer`, which provides the functionality of augmenting the system and introduces controls for the sampling decisions. 
With the keyword `params` we specify which parameters in the vector `p` are the parameters of interest, and the keyword `dt` describes the control discretization of the sampling controls ``w``.

```@example lotka_oed
oed_layer = Corleone.OEDLayer(prob, Tsit5(); params=[1,2], 
            observed = (u,p,t) -> u[1:2],
            controls = (control,),
            control_indices = [3], dt = 0.25)

ps, st = LuxCore.setup(Random.default_rng(), oed_layer)
lb, ub = Corleone.get_bounds(oed_layer)
p = ComponentArray(ps)
oed_layer(nothing, p, st)
```

To see what happens in more detail we can have a closer look at the defined controls in the `OEDLayer` and we see that there is the specified control from above, and two new sampling controls.

```@example lotka_oed
for control in oed_layer.layer.controls
    @info control.name control.t
end
```

To set up the `OptimizationProblem`, we need to define the objective function and the sampling constraints, i.e., the upper bound on the measurements. For the objective, Corleone provides different criteria. We want to use the ACriterion. It can be called in the following way:

```@example lotka_oed
crit = ACriterion()
criterion = crit(oed_layer)
criterion(p, nothing)
```

The constraints can be directly stated as linear constraints on the discretized sampling controls. For this we need to sum up the discretized controls of each measurement function.

```@example lotka_oed
nc = vcat(0, cumsum([length(x.t) for x in oed_layer.layer.controls]))

sampling_cons = let nc=nc, st=st, ax=getaxes(p), dt=0.25
    (res, p, ::Any) -> begin
        ps = ComponentArray(p, ax)
        res .= [sum(ps.controls[nc[i]+1:nc[i+1]]) * dt for i=2:3]
    end
end

sampling_cons(zeros(2), p, nothing)
```


All sampling controls are initialized with ``w=1``, therefore the constraint is evaluated as the length of the complete time horizon of the problem. We now specify that we may measure four time units and solve the problem.

```@example lotka_oed

optfun = OptimizationFunction(
    criterion, AutoForwardDiff(), cons = sampling_cons
)

optprob = OptimizationProblem(
    optfun, collect(p), lb = collect(lb), ub = collect(ub), lcons=zeros(2), ucons=[4.0,4.0]
)

uopt = solve(optprob, Ipopt.Optimizer(),
     tol = 1e-6,
     hessian_approximation = "limited-memory",
     max_iter = 300
)
```

The problem is efficiently solved. Lastly, we have a look at the solution.

```@example lotka_oed
optsol, _ = oed_layer(nothing, uopt + zero(p), st)

f = Figure()
ax = CairoMakie.Axis(f[1,1], xticks=0:2:12, title="States + control")
ax1 = CairoMakie.Axis(f[2,1], xticks=0:2:12, title="Sensitivities")
ax2 = CairoMakie.Axis(f[1,2], xticks=0:2:12, title="FIM")
ax3 = CairoMakie.Axis(f[2,2], xticks=0:2:12, title="Sampling")
[plot!(ax, optsol.t, sol) for sol in eachrow(Array(optsol))[1:2]]
[plot!(ax1, optsol.t, sol) for sol in eachrow(reduce(hcat, (optsol[Corleone.sensitivity_variables(oed_layer)[:]])))]
[plot!(ax2, optsol.t, sol) for sol in eachrow(reduce(hcat, (optsol[Corleone.fisher_variables(oed_layer)])))]
stairs!(ax, control.t, (uopt + zero(p)).controls[nc[1]+1:nc[2]])
stairs!(ax3, 0.0:0.25:11.75, (uopt + zero(p)).controls[nc[2]+1:nc[3]])
stairs!(ax3, 0.0:0.25:11.75, (uopt + zero(p)).controls[nc[3]+1:nc[4]])
CairoMakie.save("lotka_oed.svg",f) # hide
```

![](lotka_oed.svg)