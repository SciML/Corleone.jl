### A Pluto.jl notebook ###
# v0.19.46

#> [frontmatter]
#> cover = "../../src/assets/logo.png"
#> title = "ModelingToolkit Integration"
#> description = "Define and solve an optimal control problem with ModelingToolkit"
#> id = "mtk"
#> author = "Corleone contributors"
#> julia = "1.10"

using Markdown
using InteractiveUtils

# ╔═╡ 8827e882-569a-5e3f-972e-09feaaf0e110
md"""
# ModelingToolkit Integration
This tutorial models the [Lotka–Volterra fishing problem](https://mintoc.de/index.php?title=Lotka_Volterra_fishing_problem)
with [ModelingToolkit.jl](https://docs.sciml.ai/ModelingToolkit/stable/).
We solve it using single and multiple shooting, as in the
[numerical fishing example](https://docs.sciml.ai/Corleone/dev/examples/optimal_control/lotka_fishing/).

## Setup
ModelingToolkit defines the dynamics and cost. Corleone constructs the shooting
layer, ComponentArrays supplies the optimizer's parameter vector, and ForwardDiff
computes derivatives. We solve with Ipopt and plot with CairoMakie.

## Model

The states `x` and `y` are prey and predator populations. Their global initial
values are fixed because `tunable = false`. The input `u` is fishing intensity,
bounded between zero and one. Population bounds apply to lifted state values
when using multiple shooting; they do not impose continuous path constraints.
"""

# ╔═╡ 5c11a7c7-28d6-5c1f-b4de-1a9c9c16a33d
begin
    import Pkg
    Pkg.activate(Base.current_project(@__DIR__); io = devnull)
    using Corleone
    using ModelingToolkit
    using ModelingToolkit: t_nounits as t, D_nounits as D
    using OrdinaryDiffEqTsit5
    using ComponentArrays, ForwardDiff
    using Optimization
    using OptimizationMOI, Ipopt
    using LuxCore, Random
    using CairoMakie
    CairoMakie.activate!(type = "png")
end

# ╔═╡ d71e5e3f-51ca-5539-92c3-e28810cd732b
begin
    @variables begin
        x(..) = 0.5, [tunable = false, bounds = (0.0, 5.0)]
        y(..) = 0.7, [tunable = false, bounds = (0.0, 5.0)]
        u(..) = 0.0, [bounds = (0.0, 1.0), input = true]
    end
    @constants begin
        c₁ = 0.4
        c₂ = 0.2
    end
end

# ╔═╡ bbd92a13-22ec-5ab3-831c-c2e84003ce43
md"""
Fishing removes both populations at different rates.
"""

# ╔═╡ e719533b-7768-5d77-9dc4-641be747913e
begin
    eqs = [
        D(x(t)) ~ x(t) - x(t) * y(t) - c₁ * u(t) * x(t)
        D(y(t)) ~ -y(t) + x(t) * y(t) - c₂ * u(t) * y(t)
    ]
end

# ╔═╡ 87b2732f-6de9-585d-9a79-1e2be6f2c434
md"""
Minimize the accumulated squared deviations of both populations from one over
the time horizon [0, 12]. Corleone introduces a quadrature state for this integral.
"""

# ╔═╡ 435a517d-f658-5e41-b897-e0dead0f5542
begin
    costs = [
        Symbolics.Integral(t in (0.0, 12.0))(
            (x(t) - 1.0)^2 + (y(t) - 1.0)^2
        ),
    ]
    @named lotka_volterra = System(eqs, t, costs = costs)
end

# ╔═╡ 38516602-c512-56ef-ac8a-7761ae37094f
md"""
## Single Shooting

Supply the system, initial-value overrides, and the control schedule. The empty
vector keeps the system defaults. The leading zero denotes the first control
interval; this grid gives 120 piecewise constant values, one per 0.1 time units.
The integral determines the time span, so it need not be specified separately.
"""

# ╔═╡ 74987c32-0ee7-5ddb-9a5d-41bcd37e97fc
begin
    dynopt = CorleoneDynamicOptProblem(
        lotka_volterra, [],
        u(t) => 0.0:0.1:11.9,
        algorithm = Tsit5(),
    )
end

# ╔═╡ e10666a1-3a95-5838-9e3e-63cf172038be
md"""
Construct the optimization problem with the control bounds from the model.
"""

# ╔═╡ b999bb3f-1f75-5945-8e34-954a2d711f99
begin
    optprob = OptimizationProblem(dynopt, AutoForwardDiff())
end

# ╔═╡ d300baee-c7dd-5808-aa24-66e7fba9f28b
md"""
Use ForwardDiff derivatives and Ipopt's limited-memory Hessian approximation.
"""

# ╔═╡ 92cddb86-ab85-5078-8174-1c4a5cdaa6e2
begin
    uopt = solve(
        optprob, Ipopt.Optimizer(), max_iter = 1000, tol = 5.0e-6,
        hessian_approximation = "limited-memory",
        print_level = 3
    )
    @assert successful_retcode(uopt)
end

# ╔═╡ 065cc832-2b99-5341-9082-008475c9d154
md"""
Restore the parameter structure, evaluate the optimized layer, and plot it.
"""

# ╔═╡ ee81f08f-9aa2-54af-85e1-5c82409f4b4a
begin
    _, reconstruct = pack(default_vectorizer(), dynopt.ps)
    optimized_parameters = reconstruct(uopt.u)
    optsol, _ = dynopt.layer(nothing, optimized_parameters, optprob.p)
    trajectoryplot(optsol)
end

# ╔═╡ 2ff44654-e771-5a54-9b71-1588907692fc
md"""
## Multiple Shooting

Use 48 control intervals and lift the states at every interior interval boundary.
Interior shooting values remain tunable even though the global initial values
are fixed. Continuity equalities connect neighboring segments, including the
quadrature state, so the objective still accounts for the whole time horizon.
Population bounds now constrain the lifted values at every shooting point;
overly restrictive bounds can make this formulation infeasible.
"""

# ╔═╡ 59ee959d-1378-52c2-af0d-4c0e477b22f3
begin
    ms_dynopt = CorleoneDynamicOptProblem(
        lotka_volterra, [],
        u(t) => 0.0:0.25:11.75,
        algorithm = Tsit5(),
        shooting = collect(0.25:0.25:11.75),
        problem_kwargs = (; abstol = 1.0e-9, reltol = 1.0e-9)
    )
end

# ╔═╡ 6f937c3c-3ab1-5370-ad7a-512bd9d1a83e
md"""
Corleone appends the shooting equalities to the optimization constraints.
"""

# ╔═╡ 3911f881-0f60-56ea-a67a-20d2492a18b8
begin
    ms_optprob = OptimizationProblem(ms_dynopt, AutoForwardDiff())

    ms_result = solve(
        ms_optprob,
        Ipopt.Optimizer(),
        tol = 1.0e-6,
        hessian_approximation = "limited-memory",
        max_iter = 300,
        print_level = 3
    )
    @assert successful_retcode(ms_result)
end

# ╔═╡ b17b3ed5-8cc4-5530-a5c8-530c75e5bbc7
md"""
Reconstruct and plot the multiple-shooting solution in the same way.
"""

# ╔═╡ 5667cc29-9dd7-5bc6-a58e-29ceba34b7cc
begin
    _, reconstruct_ms = pack(default_vectorizer(), ms_dynopt.ps)
    ms_optimized_parameters = reconstruct_ms(ms_result.u)
    ms_trajectory, _ = ms_dynopt.layer(nothing, ms_optimized_parameters, ms_optprob.p)
    trajectoryplot(ms_trajectory)
end

# ╔═╡ 5bee8d10-6b82-5e99-b68b-e7a5d4a34d3b
md"""
Check continuity after optimization. Small residuals connect the lifted physical
states and quadrature; integration and optimizer tolerances are separate controls
on the numerical result. The two formulations use different control grids, so
their objective values are not an exact discretization-to-discretization check.
"""

# ╔═╡ f76c974f-f11e-5c5a-a039-2e0123563fc3
begin
    maximum(abs, shooting_constraints(ms_trajectory))
end

# ╔═╡ Cell order:
# ╟─8827e882-569a-5e3f-972e-09feaaf0e110
# ╠═5c11a7c7-28d6-5c1f-b4de-1a9c9c16a33d
# ╠═d71e5e3f-51ca-5539-92c3-e28810cd732b
# ╟─bbd92a13-22ec-5ab3-831c-c2e84003ce43
# ╠═e719533b-7768-5d77-9dc4-641be747913e
# ╟─87b2732f-6de9-585d-9a79-1e2be6f2c434
# ╠═435a517d-f658-5e41-b897-e0dead0f5542
# ╟─38516602-c512-56ef-ac8a-7761ae37094f
# ╠═74987c32-0ee7-5ddb-9a5d-41bcd37e97fc
# ╟─e10666a1-3a95-5838-9e3e-63cf172038be
# ╠═b999bb3f-1f75-5945-8e34-954a2d711f99
# ╟─d300baee-c7dd-5808-aa24-66e7fba9f28b
# ╠═92cddb86-ab85-5078-8174-1c4a5cdaa6e2
# ╟─065cc832-2b99-5341-9082-008475c9d154
# ╠═ee81f08f-9aa2-54af-85e1-5c82409f4b4a
# ╟─2ff44654-e771-5a54-9b71-1588907692fc
# ╠═59ee959d-1378-52c2-af0d-4c0e477b22f3
# ╟─6f937c3c-3ab1-5370-ad7a-512bd9d1a83e
# ╠═3911f881-0f60-56ea-a67a-20d2492a18b8
# ╟─b17b3ed5-8cc4-5530-a5c8-530c75e5bbc7
# ╠═5667cc29-9dd7-5bc6-a58e-29ceba34b7cc
# ╟─5bee8d10-6b82-5e99-b68b-e7a5d4a34d3b
# ╠═f76c974f-f11e-5c5a-a039-2e0123563fc3
