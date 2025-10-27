# Multiple shooting example

This page shows how to use multiple shooting for solving optimal control problems and also how to use the initialization methods for the shooting nodes. Here, we use the well-known Fuller's problem. The relaxed problem reads

```math
\begin{array}{llcl}
 \displaystyle \min_{x, u} & \int_{0}^{1} x_0^2 & \; \mathrm{d} t \\[1.5ex]
 \mbox{s.t.} & \dot{x}_0 & = & x_1, \\
 & \dot{x}_1 & = & 1 - 2 \; u, \\[1.5ex]
 & x(0) &=& x_S, \\
 & x(t_f) &=& x_T, \\
 & u(t) &\in&  [0, 1].
\end{array} 
```
with ``x_S = x_T = [0.01, 0]^\top``. 

```@example fuller
using Corleone
using Corleone.LuxCore, Corleone.ComponentArrays
using OrdinaryDiffEqTsit5
using SciMLSensitivity
using Random, CairoMakie
using Optimization, OptimizationMOI, Ipopt


function fuller(du, u, p, t)
    du[1] = u[2]
    du[2] = 1.0 - 2.0 * p[1]
    du[3] = u[1]^2
end

tspan = (0.0, 1.0)
u0 = [1e-2, 0.0, 0.0]
p = [0.5]

prob =  ODEProblem(fuller, u0, tspan, p,
        abstol=1e-8, reltol=1e-8
)
```

Note that the function `fuller` is now written as an in-place function, mutating the first argument `du`, and again, the control acts on the system via the parameter vector p, this time in the first position.
Next, we define the control discretization with a grid size of ``0.01``.

```@example fuller
dt = 0.01
cgrid = collect(0.0:dt:1.0)[1:end-1]
control = ControlParameter(
    cgrid, name = :u, controls = ones(length(cgrid)), bounds = (0,1)
)
```

We can now define the `MultipleShootingLayer`.

```@example fuller

shooting_points = [0.0, 0.2, 0.4, 0.6, 0.8]
layer = Corleone.MultipleShootingLayer(prob, Tsit5(), [1], (control,), shooting_points)

ps, st = LuxCore.setup(Random.default_rng(), layer)
p = ComponentArray(ps)
lb, ub = Corleone.get_bounds(layer)
layer # hide 
```

Let's have a look at the solution of the system.

```@example fuller

sols, _ = layer(nothing, p, st)

f = Figure()
titles = ["x₀", "x₁", "x₂", "u"]
for i in 1:4
    ax = f[i, 1] = CairoMakie.Axis(f, title=titles[i])
    for (j,sol) in enumerate(sols)
        plt = i == 4 ? stairs! : lines!
        plt(ax, sol.t, Array(sol)[i, :],)
    end
end
CairoMakie.save("fuller_initial.svg",f) # hide
```

![](fuller_initial.svg)

To get a continuous trajectory, we need to add the continuity conditions to the problem, i.e., equality constraints stating that the solution at the end of one shooting interval equals the shooting node of the subsequent shooting interval. The function to evaluate this constraint is already provided by Corleone.

```@example fuller

continuity = Corleone.get_shooting_constraints(layer)

continuity_init = continuity(sols, p)
nc = length(continuity_init)
continuity_init
```

There are also different initialization methods provided in Corleone to initialize the shooting nodes in a flexible way. One method is to use a forward solve of the system and use the solution at the shooting time points for initialization, leading to a continuous initial trajectory.

```@example fuller
p_fwd, _ = ForwardSolveInitialization()(Random.default_rng(),layer)
sols_fwd, _ = layer(nothing, p_fwd, st)
continuity(sols_fwd, p_fwd)
```

Now, we need to state the terminal constraint and set up and solve the optimization problem.

```@example fuller
objective = let layer = layer, st = st, ax = getaxes(p)
    (p, ::Any) -> begin
        ps = ComponentArray(p, ax)
        sols, _ = layer(nothing, ps, st)
        last(sols)[:x₃][end]
    end
end


cons = let layer = layer, st = st, ax = getaxes(p), continuity=continuity
    (res, p, ::Any) -> begin
        ps = ComponentArray(p, ax)
        sols, _ = layer(nothing, ps, st)
        matching_constraint = continuity(sols, ps)
        res .= vcat(last(sols)[:x₁][end], last(sols)[:x₂][end], matching_constraint)
    end
end


optfun = OptimizationFunction(
    objective, AutoForwardDiff(), cons=cons
)

optprob = OptimizationProblem(
    optfun, collect(p), lb = collect(lb), ub = collect(ub),
    lcons = vcat([1e-2, 0.0], zeros(nc)),
    ucons = vcat([1e-2, 0.0], zeros(nc))
)

uopt = solve(optprob, Ipopt.Optimizer(),
    tol = 1e-12,
    hessian_approximation = "limited-memory",
    max_iter = 250
)
```

The problem can be efficiently solved. Let's investigate the solution.

```@example fuller

sols, _ = layer(nothing, uopt + zero(p), st)

f = Figure()
for i in 1:4
    ax = f[i, 1] = CairoMakie.Axis(f, title=titles[i])
    for (j,sol) in enumerate(sols)
        plt = i == 4 ? stairs! : lines!
        plt(ax, sol.t, Array(sol)[i, :],)
    end
end
CairoMakie.save("fuller_solution.svg",f) # hide
```

![](fuller_solution.svg)
