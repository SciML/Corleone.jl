### A Pluto.jl notebook ###
# v0.19.46

#> [frontmatter]
#> cover = "../../src/assets/tutorials/tracking.svg"
#> title = "Track a target with linear quadratic control"
#> description = "Build a shooting layer, optimize a piecewise constant control, and inspect state trajectories."
#> id = "linear_quad_reg"
#> author = "Corleone contributors"
#> julia = "1.10"

using Markdown
using InteractiveUtils

# ╔═╡ c41b1601-7660-504e-8d8d-5be033388cb0
md"""
# Track a target with linear quadratic control

We control two linear states with a shared input. The cost penalizes the first
state's deviation from three and the control effort:

```math
\begin{aligned}
\min_u & \int_0^{12} \bigl(10 (x_1 - 3)^2 + 0.1 u^2\bigr) \, dt \\
\text{s.t.} \quad & \dot{x}_i = -x_i + u, \quad i = 1, 2, \\
& x_1(0)=1,\quad x_2(0)=2.
\end{aligned}
```

## Setup

Use Corleone to define the shooting layer and optimization model.

The other packages provide:

- Layer initialization: `LuxCore` and `Random`
- ODE integration: `OrdinaryDiffEqTsit5`
- Optimization: `Optimization`, `OptimizationLBFGSB`, and `ComponentArrays`
- Objective gradients: `ForwardDiff`
- Plotting: `CairoMakie`

## LQR Dynamics

The first two states follow the linear dynamics. The third accumulates the
running cost, which depends on the first state and the control.
"""

# ╔═╡ 99c251d7-33e2-54f2-97bb-7b473a28640a
begin
    import Pkg
    Pkg.activate(Base.current_project(@__DIR__); io = devnull)
    using Corleone
    using LuxCore
    using Random
    using OrdinaryDiffEqTsit5
    using Optimization
    using OptimizationLBFGSB
    using ForwardDiff
    using ComponentArrays
    using SymbolicIndexingInterface
    using CairoMakie
    CairoMakie.activate!(type = "png")
end

# ╔═╡ 71d7aaa4-ebed-52b3-a508-3ee255bdba6c
begin
    function lqr_dynamics(x, p, t)
        a, b, u = p
        du = a .* x[1:(end - 1)] .+ b .* u
        costs = 10.0 .* (x[1] .- 3.0) .^ 2 .+ 0.1 .* u .^ 2
        return vcat(du, costs)
    end

    tspan = (0.0, 12.0)
    u0 = [1.0, 2.0, 0.0]
    p0 = [-1.0, 1.0, 0.0]
    lqr_problem = ODEProblem(lqr_dynamics, u0, tspan, p0)
end

# ╔═╡ 25652c4b-1682-5790-b7d1-14247c18e5a2
md"""
## Optimal Control Problem

Make the third parameter a piecewise constant control and keep `a` and `b` fixed.
"""

# ╔═╡ 82d95187-9a7c-5604-964c-f8a61f368ffa
begin
    control_grid = collect(0.1:0.1:11.9)
    predictor = ShootingLayer(
        lqr_problem, Tsit5(),
        Tunable(3; kind = Val(:parameter), default = (rng, n) -> rand(rng, n)) => control_grid,
        Constant(1; kind = Val(:parameter)) => nothing,
        Constant(2; kind = Val(:parameter)) => nothing;
        controls = [3], quadratures = [3],
    )
end

# ╔═╡ 4956bac6-c423-5903-a3a7-d9fcb256a149
md"""
The 119 switching times define 120 control intervals. Initial control values
are sampled uniformly from [0, 1). The fixed parameters are stored in Lux state.

Preview the trajectory with the initialized control.
"""

# ╔═╡ 2ce62521-7496-5c09-aabe-6507af4c42f5
begin
    ps, st = LuxCore.setup(MersenneTwister(42), predictor)
    solution, _ = predictor(nothing, ps, st)
    trajectoryplot(solution)
end

# ╔═╡ 1d085b6b-3238-5690-bc59-a0ed795390f2
md"""
## Optimize

Use the accumulated cost `u₃` at the final time as the objective. Retain the
seeded parameters and state from the preview as the optimization's initial guess.
"""

# ╔═╡ 208ea22e-9d18-53d6-8c30-fa4ffe399bac
begin
    dynprob = CorleoneDynamicOptProblem(
        predictor, :(u₃(12.0)); ps, st
    )
end

# ╔═╡ 07ca95f1-96cc-575b-a73d-709ec0a2ffa2
md"""
OptimizationProblem packs the parameters and uses ForwardDiff for gradients.
"""

# ╔═╡ df3ee580-4d42-5edd-8110-ad060bdd10c6
begin
    opt = OptimizationProblem(dynprob, AutoForwardDiff(); function_kwargs = (;))
    usol = solve(opt, OptimizationLBFGSB.LBFGSB())
    @assert successful_retcode(usol)
end

# ╔═╡ a64e8d0a-bbd4-5202-8fe6-3630b77e0401
md"""
Reconstruct the optimized parameters and plot the trajectory.
"""

# ╔═╡ 9054627c-a9d0-5991-9deb-eead2628d2c3
begin
    _, reconstruct = pack(default_vectorizer(), dynprob.ps)
    optimized_parameters = reconstruct(usol.u)
    optimized_solution, _ = predictor(nothing, optimized_parameters, opt.p)
    trajectoryplot(optimized_solution)
end

# ╔═╡ 0f692f55-c84c-5dc9-8349-a35f8771bcda
md"""
The optimized first state approaches the target of three. The second state
follows the same input but does not enter the objective. This example uses an
unbounded piecewise-constant control; the random initial guesses are not bounds.
"""

# ╔═╡ 8b4a86a2-cfd4-5551-91c4-2ce0c8755e4c
begin
    usol.objective
end

# ╔═╡ Cell order:
# ╟─c41b1601-7660-504e-8d8d-5be033388cb0
# ╠═99c251d7-33e2-54f2-97bb-7b473a28640a
# ╠═71d7aaa4-ebed-52b3-a508-3ee255bdba6c
# ╟─25652c4b-1682-5790-b7d1-14247c18e5a2
# ╠═82d95187-9a7c-5604-964c-f8a61f368ffa
# ╟─4956bac6-c423-5903-a3a7-d9fcb256a149
# ╠═2ce62521-7496-5c09-aabe-6507af4c42f5
# ╟─1d085b6b-3238-5690-bc59-a0ed795390f2
# ╠═208ea22e-9d18-53d6-8c30-fa4ffe399bac
# ╟─07ca95f1-96cc-575b-a73d-709ec0a2ffa2
# ╠═df3ee580-4d42-5edd-8110-ad060bdd10c6
# ╟─a64e8d0a-bbd4-5202-8fe6-3630b77e0401
# ╠═9054627c-a9d0-5991-9deb-eead2628d2c3
# ╟─0f692f55-c84c-5dc9-8349-a35f8771bcda
# ╠═8b4a86a2-cfd4-5551-91c4-2ce0c8755e4c
