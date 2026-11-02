### A Pluto.jl notebook ###
# v0.19.46

#> [frontmatter]
#> cover = "../../src/assets/tutorials/predator-prey.svg"
#> title = "Control predator–prey dynamics"
#> description = "Progress from single to multiple shooting with bounded fishing and continuity constraints."
#> id = "lotka_fishing"
#> author = "Corleone contributors"
#> julia = "1.10"

using Markdown
using InteractiveUtils

# ╔═╡ a25adae1-ca05-58ae-90bc-ed399054fe9e
md"""
# Control predator–prey dynamics
This example is based on the [Lotka-Volterra fishing problem](https://mintoc.de/index.php?title=Lotka_Volterra_fishing_problem).

## Setup

Use Corleone to define the shooting layers and optimization models.

The other packages provide:

- Layer initialization: `LuxCore` and `Random`
- ODE integration: `OrdinaryDiffEqTsit5`
- Optimization: `Optimization`, `OptimizationMOI`, `Ipopt`, and `ComponentArrays`
- Objective and constraint gradients: `ForwardDiff`
- Plotting: `CairoMakie`

## Lotka–Volterra Dynamics

The first two states are prey and predator populations. The third accumulates
their squared deviations from one. The second parameter is fishing intensity;
the first and third are fixed interaction rates.

```math
\begin{aligned}
\dot{x} &= x-xy-0.4vx, & x(0)&=0.5, \\
\dot{y} &= -y+xy-0.2vy, & y(0)&=0.7, \\
J &= \int_0^{12}\bigl((x-1)^2+(y-1)^2\bigr)\,dt, & 0&\le v\le1.
\end{aligned}
```
Here `v` is the fishing control, stored as `p[2]` in the numerical model.
"""

# ╔═╡ a16611f8-e2ef-54f4-b663-c7f038a22ef5
begin
    import Pkg
    Pkg.activate(Base.current_project(@__DIR__); io = devnull)
    using Corleone
    using LuxCore
    using Random
    using OrdinaryDiffEqTsit5
    using SymbolicIndexingInterface
    using Optimization
    using OptimizationMOI
    using Ipopt
    using ComponentArrays
    using ForwardDiff
    using CairoMakie
    CairoMakie.activate!(type = "png")
end

# ╔═╡ c0315e1e-0dbd-5c03-aea7-c6ddfd982f21
begin
    function lotka_dynamics(du, u, p, t)
        du[1] = u[1] - p[1] * prod(u[1:2]) - 0.4 * p[2] * u[1]
        du[2] = -u[2] + p[3] * prod(u[1:2]) - 0.2 * p[2] * u[2]
        return du[3] = (u[1] - 1.0)^2 + (u[2] - 1.0)^2
    end
    tspan = (0.0, 12.0)
    u0 = [0.5, 0.7, 0.0]
    p0 = [1.0, 0.0, 1.0]
    prob = ODEProblem(lotka_dynamics, u0, tspan, p0; abstol = 1.0e-9, reltol = 1.0e-9)
end

# ╔═╡ b42dba3f-88eb-53c4-b84c-1090fd0ddc34
md"""
## Single Shooting Approach

Make fishing intensity tunable between zero and one, and keep the two
interaction rates fixed. The 119 switching times define 120 control intervals.
The initial state is taken from the ODE problem.
"""

# ╔═╡ 2d53f7d2-74e5-5e6d-9dda-e3f9dc9924ed
begin
    cgrid = collect(0.1:0.1:11.9)
    control = Tunable(
        2; kind = Val(:parameter), default = 1.0,
        lowerbound = 0.0, upperbound = 1.0
    ) => cgrid
    physical_parameters = (
        Constant([3, 1]; kind = Val(:parameter)) => nothing,
    )
    layer = ShootingLayer(
        prob, Tsit5(), control, physical_parameters...;
        controls = [2], quadratures = [3]
    )
end

# ╔═╡ 307dbb88-c781-5621-aa11-dafa5270587c
md"""
Minimize the terminal accumulated cost. The model initializes parameters and
state, then prepares the objective. OptimizationProblem applies the control bounds
from the annotation and uses ForwardDiff for gradients.
"""

# ╔═╡ daf804c2-386d-5367-8c48-6c809669fb14
begin
    dynprob = CorleoneDynamicOptProblem(
        layer, :(u₃(12.0))
    )
    opt = OptimizationProblem(dynprob, AutoForwardDiff(); function_kwargs = (;))
    uopt = solve(
        opt,
        Ipopt.Optimizer(),
        tol = 1.0e-6,
        hessian_approximation = "limited-memory",
        max_iter = 300,
        print_level = 3
    )
    @assert successful_retcode(uopt)
end

# ╔═╡ 30466bbb-0021-59bc-a72c-520b7041abca
md"""
Reconstruct the optimized parameters and plot the trajectory.
"""

# ╔═╡ 04963cbe-446f-5eb7-aa73-bbf50bf8a216
begin
    _, reconstruct = pack(default_vectorizer(), dynprob.ps)
    optimized_parameters = reconstruct(uopt.u)
    optsol, _ = layer(nothing, optimized_parameters, opt.p)
    trajectoryplot(optsol)
end

# ╔═╡ cb9da73d-85f4-5f3b-ad3f-6e590bdd637f
md"""
## Multiple Shooting

Lift all three states at interior shooting points. Matching constraints enforce
continuity, including the accumulated cost, so its final value covers the whole
time horizon. Full-state resets allow independent blocks with `EnsembleThreads()`.
Start Julia with `--threads=4` to use four threads.

Start with control intervals of length 0.25. The shooting points at 3.4 and 6.1
lie inside intervals; 9.75 already coincides with a control switch.
"""

# ╔═╡ 2fb89631-1b71-5e4b-9f7c-3dfe92854b17
begin
    shooting_points = [3.4, 6.1, 9.75]
    initial_conditions = (
        Tunable(i; kind = Val(:variable), lowerbound = 0.0, upperbound = 2.0) => shooting_points for i in 1:3
    )
    mslayer = ShootingLayer(
        prob, Tsit5(), first(control) => collect(0.25:0.25:11.75), physical_parameters...,
        initial_conditions...; ensemblealg = EnsembleThreads(),
        controls = [2], quadratures = [3]
    )
end

# ╔═╡ a6bb979a-2be2-5786-a4d9-aedfc7de6bd4
md"""
Initialize controls to one and lifted states to the corresponding ODE initial values.
"""

# ╔═╡ 4cb9aa13-ad4e-582c-bf45-32708da06cdb
begin
    initial_msps, initial_msst = LuxCore.setup(MersenneTwister(42), mslayer)
end

# ╔═╡ 67a0bd8b-acd4-52b3-a30f-574a70ddd175
md"""
Split the control intervals at 3.4 and 6.1, duplicating their current values.
"""

# ╔═╡ eb3be474-ff31-5a72-809a-da4bb02ea13b
begin
    msps, msst = add_parameter_segments!(mslayer, initial_msps, initial_msst)
end

# ╔═╡ ed621753-7eb0-5d96-8af5-1f5fb06f24bc
md"""
Plot the initial segments and their reset jumps.
"""

# ╔═╡ 8975d6ba-a6d3-5c1a-ae59-0bb3fb245656
begin
    mssol, _ = mslayer(nothing, msps, msst)
    trajectoryplot(mssol, show_legend = true, whiskerwidth = 10)
end

# ╔═╡ 8b8d432b-37d3-56f8-b8e2-46f7aa6b4f23
md"""
Reuse the expanded parameters and state. State continuity and matching control
values at inserted switches are included as equality constraints.
"""

# ╔═╡ 762bff67-9d7e-5ce3-bc56-bfc5f8b47c26
begin
    ms_dynprob = CorleoneDynamicOptProblem(
        mslayer, :(u₃(12.0)), ps = msps, st = msst
    )

    ms_opt = OptimizationProblem(ms_dynprob, AutoForwardDiff(); function_kwargs = (;))
    ms_uopt = solve(
        ms_opt,
        Ipopt.Optimizer(),
        tol = 1.0e-6,
        hessian_approximation = "limited-memory",
        max_iter = 1000,
        print_level = 5
    )
    @assert successful_retcode(ms_uopt)
end

# ╔═╡ dcd67608-668b-5cb8-ae57-63e8a1b5a7fb
md"""
Reconstruct the optimized parameters and plot the trajectory.
"""

# ╔═╡ c12c5a33-a0b6-55d9-872a-ab109c2c15d4
begin
    _, reconstruct_ms = pack(default_vectorizer(), ms_dynprob.ps)
    ms_optimized_parameters = reconstruct_ms(ms_uopt.u)
    ms_optsol, _ = mslayer(nothing, ms_optimized_parameters, ms_opt.p)
    trajectoryplot(ms_optsol)
end

# ╔═╡ 4e09707e-996e-5570-9aaf-764a6834903e
md"""
Matching residuals measure feasibility of the lifted states and inserted control
intervals. The single- and multiple-shooting problems use different control
grids, and this multiple-shooting model also bounds its lifted states (including
the quadrature) by two. Their objective values need not coincide.
"""

# ╔═╡ fedb9134-bd4f-5288-86dc-79eb60001d09
begin
    maximum(abs, shooting_constraints(ms_optsol))
end

# ╔═╡ Cell order:
# ╟─a25adae1-ca05-58ae-90bc-ed399054fe9e
# ╠═a16611f8-e2ef-54f4-b663-c7f038a22ef5
# ╠═c0315e1e-0dbd-5c03-aea7-c6ddfd982f21
# ╟─b42dba3f-88eb-53c4-b84c-1090fd0ddc34
# ╠═2d53f7d2-74e5-5e6d-9dda-e3f9dc9924ed
# ╟─307dbb88-c781-5621-aa11-dafa5270587c
# ╠═daf804c2-386d-5367-8c48-6c809669fb14
# ╟─30466bbb-0021-59bc-a72c-520b7041abca
# ╠═04963cbe-446f-5eb7-aa73-bbf50bf8a216
# ╟─cb9da73d-85f4-5f3b-ad3f-6e590bdd637f
# ╠═2fb89631-1b71-5e4b-9f7c-3dfe92854b17
# ╟─a6bb979a-2be2-5786-a4d9-aedfc7de6bd4
# ╠═4cb9aa13-ad4e-582c-bf45-32708da06cdb
# ╟─67a0bd8b-acd4-52b3-a30f-574a70ddd175
# ╠═eb3be474-ff31-5a72-809a-da4bb02ea13b
# ╟─ed621753-7eb0-5d96-8af5-1f5fb06f24bc
# ╠═8975d6ba-a6d3-5c1a-ae59-0bb3fb245656
# ╟─8b8d432b-37d3-56f8-b8e2-46f7aa6b4f23
# ╠═762bff67-9d7e-5ce3-bc56-bfc5f8b47c26
# ╟─dcd67608-668b-5cb8-ae57-63e8a1b5a7fb
# ╠═c12c5a33-a0b6-55d9-872a-ab109c2c15d4
# ╟─4e09707e-996e-5570-9aaf-764a6834903e
# ╠═fedb9134-bd4f-5288-86dc-79eb60001d09
