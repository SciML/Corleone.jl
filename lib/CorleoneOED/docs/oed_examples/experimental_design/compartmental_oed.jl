### A Pluto.jl notebook ###
# v0.19.46

#> [frontmatter]
#> cover = "../../../../../docs/src/assets/logo.png"
#> title = "Discrete measurements for a compartmental model"
#> description = "Allocate measurement weights at candidate sampling times."
#> id = "compartmental_oed"
#> author = "Christoph Plate; Carl Julius Martensen; Sebastian Sager"
#> julia = "1.10"

using Markdown
using InteractiveUtils

# ╔═╡ 61574761-9856-52dc-acf3-c5f9d5c8f732
md"""
# Compartmental OED
Estimate three parameters by observing one state. We optimize a relaxed
selection of six measurements from 19 candidate times, following the
[compartmental OED benchmark](https://mintoc.de/index.php?title=Compartmental_OED).
The nominal vector is `p = [θ₁, θ₂, θ₃]`. All three parameters are estimated but
fixed as design inputs; only measurement weights are optimized. We use no
physical control or shooting resets, hence no shooting matching equalities.

ComponentArrays supplies packing; Optimization/OptimizationMOI and Ipopt solve
the continuous relaxation; CairoMakie plots event weights. A forward OED solve
itself needs none of those optional optimization/plotting packages.
"""

# ╔═╡ 2177c5c9-bb60-5f61-99cf-54d1aac6e884
begin
    import Pkg
    Pkg.activate(Base.current_project(@__DIR__); io = devnull)
    using CorleoneOED, LuxCore, Random, OrdinaryDiffEqTsit5, LinearAlgebra
    using Optimization, OptimizationMOI, Ipopt, ComponentArrays, CairoMakie
    using SymbolicIndexingInterface: parameter_values_at_time, parameter_index
    CairoMakie.activate!(type = "png")
end

# ╔═╡ acf4a2d2-f737-51b1-8967-16bac60b2d7f
begin
    function compartmental(du, u, p, t)
        θ1, θ2, θ3 = p
        du[1] = θ3 * (-θ1 * exp(-θ1 * t) + θ2 * exp(-θ2 * t))
        return nothing
    end
    problem = ODEProblem(compartmental, [0.0], (0.0, 50.0), [0.05884, 4.298, 21.8])
    measurement_points = [
        0.0, 0.166, 0.333, 0.5, 0.666, 1.0, 1.5, 2.0, 2.5, 3.0,
        4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 24.0, 30.0, 48.0,
    ]
    oed = OEDLayer(
        problem, Tsit5(); params = [1, 2, 3], measurements = (
            DiscreteMeasurement(:sample, measurement_points, (x, p, t) -> x[1]),
        )
    )
    ps, st = LuxCore.setup(Xoshiro(1), oed)
end

# ╔═╡ 2de14d8d-561c-5bad-8681-4002099d8f88
md"""
## Solve the relaxed design
Physical parameters remain fixed. Each candidate time has a weight in `[0,1]`;
the total weight is bounded by six. Fractional weights are allowed in this solve.
Unlike a continuous density, this budget counts event weights rather than time.
The first point at `0.0` is supported even though this model's state sensitivity
is zero there; endpoint measurements are not implicitly removed. `DCriterion`
returns `1/det(F)`, without a logarithm or dimension normalization.
"""

# ╔═╡ db08beb6-3731-5c10-bdd8-bc167a02fc71
begin
    optprob = OptimizationProblem(oed, DCriterion(); ps, st, M = [6.0])
    result = solve(
        optprob, Ipopt.Optimizer(); tol = 1.0e-8,
        hessian_approximation = "limited-memory", max_iter = 100, print_level = 0
    )
    @assert CorleoneOED.SciMLBase.successful_retcode(result)
    @assert isfinite(result.objective)
    _, reconstruct = pack(default_vectorizer(), ps)
    optimal_ps = reconstruct(result.u)
    trajectory, _ = oed(nothing, optimal_ps, st)
    F = fisher_information(oed, trajectory)
    optimal_budget = only(sampling_sums(oed, trajectory))
    @assert isposdef(Symmetric(F))
    @assert optimal_budget <= 6.0 + 1.0e-5
    @assert isapprox(result.objective, DCriterion()(F); rtol = 1.0e-5)
    (; termination = result.retcode, objective = result.objective, budget = optimal_budget)
end

# ╔═╡ 43cadd31-6260-5701-9603-8cc4537acb56
md"""
Evaluate weights from the current parameter schedule at actual event times.
The saved augmented states contain sensitivities/information, not weight histories.
"""

# ╔═╡ fcb7993d-0819-526b-872b-5b1030e26c8b
begin
    weights = [
        parameter_values_at_time(trajectory, trajectory, t)[parameter_index(trajectory, :sample)]
            for t in measurement_points
    ]
    figure = Figure()
    axis = CairoMakie.Axis(figure[1, 1]; xlabel = "Time", ylabel = "Measurement weight")
    scatter!(axis, measurement_points, weights)
    figure
end

# ╔═╡ 8c6c921b-90e8-5e26-b9b5-763618ff24ca
md"""
Optimized relaxed weights at the 19 candidate times. A point near one is a heavily
weighted event; fractional values are not rounded here. Successful local solver
termination and feasible budgets do not prove a globally optimal integer design.

## Integer sampling decisions
To require binary selections, construct the problem below and choose an
integer-capable optimizer. Ipopt solves the continuous relaxation only.
"""

# ╔═╡ bfe38c8a-0b05-56f7-931f-9683f8f05c8d
begin
    integer_problem = OptimizationProblem(oed, DCriterion(); ps, st, M = [6.0], integer_weights = true)
    @assert sum(integer_problem.int) == length(measurement_points)
end

# ╔═╡ 991c4462-a07e-593d-b65f-ba3a2baa7eff
md"""
This creates a binary-weight problem description; it does not solve it. Passing
the integer mask to a continuous solver does not enforce binary choices.
"""

# ╔═╡ Cell order:
# ╟─61574761-9856-52dc-acf3-c5f9d5c8f732
# ╠═2177c5c9-bb60-5f61-99cf-54d1aac6e884
# ╠═acf4a2d2-f737-51b1-8967-16bac60b2d7f
# ╟─2de14d8d-561c-5bad-8681-4002099d8f88
# ╠═db08beb6-3731-5c10-bdd8-bc167a02fc71
# ╟─43cadd31-6260-5701-9603-8cc4537acb56
# ╠═fcb7993d-0819-526b-872b-5b1030e26c8b
# ╟─8c6c921b-90e8-5e26-b9b5-763618ff24ca
# ╠═bfe38c8a-0b05-56f7-931f-9683f8f05c8d
# ╟─991c4462-a07e-593d-b65f-ba3a2baa7eff
