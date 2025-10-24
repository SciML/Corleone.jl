# Simple optimal control example

This page shows how to formulate an easy optimal control problem in Corleone. The problem is
the well-known Lotka-Volterra fishing problem, which reads

```math
\begin{align}
    \begin{split}
	  \displaystyle \min_{x, u} 
   & \displaystyle \int_{0}^{12} (x_1(t)-1)^2 + (x_2(t)-1)^2 \mathrm{d}t  \\[1.5ex]
	  \mbox{s.t.} \   
	  & \dot{x}_1(t)  = x_1(t) - p_1 x_1(t)x_2(t) - c_1 u(t) x_1(t)  \\
	  & \dot{x}_2(t)  =  -x_2(t) + p_2 x_1(t)x_2(t) - c_2  u(t) x_2(t)\\
	  & x(0)  =  [0.5, 0.7], \\
	  &u(t)  \in  [0,1],
    \end{split}
    \tag{Lotka}
\end{align}
```
with ``c_1=0.4`` and ``c_2=0.2`` being constants.

The goal is to drive both species ``x_1`` and ``x_2`` into the steady state `[1.0,1.0]` using the control ``u``.

The first step is to define the underlying differential equation problem.

```@example lotka
using Corleone
using Corleone.LuxCore, Corleone.ComponentArrays
using OrdinaryDiffEqTsit5
using SciMLSensitivity
using Random, CairoMakie
using Optimization, OptimizationMOI, Ipopt

function lotka(u, p, t)
    return [u[1] - p[1] * prod(u[1:2]) - 0.4 * p[3] * u[1];
            -u[2] + p[2] * prod(u[1:2]) - 0.2 * p[3] * u[2];
            (u[1]-1.0)^2 + (u[2] - 1.0)^2]
end

tspan = (0., 12.)
u0 = [0.5, 0.7, 0.]
p0 = [1.0, 1.0, 0.0]

prob = ODEProblem(lotka, u0, tspan, p0)
```

Note that the control ``u`` is defined as the third index in the parameter vector `p`. 
Next, we define the discretization of the control ``u`` as piecewise constant on a suitable time grid.
We use an equidistant control discretization of grid size `0.1`. 

```@example lotka

control_grid = collect(0.0:0.1:11.9)
control = ControlParameter(
    control_grid, name = :fishing, bounds=(0.0,1.0)
)
```

Now, we need to bring the control definition and the `ODEProblem` together, defining 
a `SingleShootingLayer` that can be called to simulate the system with discretized values for the
controls. For this, we need to specify the index of the control in the parameter vector.
We can also directly get all degrees of freedom of the layer and their bounds and simulate the system
once.

```@example lotka
layer = Corleone.SingleShootingLayer(prob, Tsit5(), [3], (control,))
ps, st = LuxCore.setup(Random.default_rng(), layer)
p = ComponentArray(ps)
lb, ub = Corleone.get_bounds(layer)

layer(nothing, ps, st)
```


Setting up the optimization problem is easy. We need to define the objective function and set
up the `OptimizationProblem`

```@example lotka
loss = let layer = layer, st = st, ax = getaxes(p)
    (p, ::Any) -> begin
        ps = ComponentArray(p, ax)
        sols, _ = layer(nothing, ps, st)
        sols[:x₃][end]
    end
end

optfun = OptimizationFunction(
    loss, AutoForwardDiff(),
)

optprob = OptimizationProblem(
    optfun, collect(p), lb = collect(lb), ub = collect(ub)
)

uopt = solve(optprob, Ipopt.Optimizer(),
     tol = 1e-6,
     hessian_approximation = "limited-memory",
     max_iter = 300
)
```

The problem is efficiently solved. Lastly, we have a look at the solution.

```@example lotka
optsol, _ = layer(nothing, uopt + zero(p), st) 

f = Figure()
ax = CairoMakie.Axis(f[1,1])
[lines!(ax, optsol.t, optsol[x], label = string(x)) for x in [:x₁, :x₂]]
f[1, 2] = Legend(f, ax, "States", framevisible = false)
ax1 = CairoMakie.Axis(f[2,1])
stairs!(ax1, optsol.t, optsol[:u₁], label = "u₁")
f[2, 2] = Legend(f, ax1, "Controls", framevisible = false)
CairoMakie.save("lotka.svg",f) # hide
```

![](lotka.svg)