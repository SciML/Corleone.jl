# Category: episodical
# Lineages adjust expression burden during a nutrient-response episode.
# GenomeSize dynamic-game case specification.
using CorleoneGame
module GenomeSize
    using CorleoneGame

    Base.@kwdef struct Parameters
        initial::NTuple{2, Float64} = (0.2, 0.3)
        T::Float64 = 8.0
        mumax::NTuple{2, Float64} = (1.0, 0.9)
        K::NTuple{2, Float64} = (1.0, 1.4)
        d::NTuple{2, Float64} = (0.1, 0.12)
        # Expression burden inhibits growth noticeably, so expression must be timed.
        alpha::NTuple{2, Float64} = (0.8, 1.0)
        g::NTuple{2, Float64} = (0.4, 0.5)
        dg::Float64 = 0.2
        D::Float64 = 0.3
        Sin::Float64 = 2.0
        Y::NTuple{2, Float64} = (0.8, 0.75)
        x0::NTuple{4, Float64} = (0.5, 0.45, 0.2, 2.0)
        w_A::NTuple{3, Float64} = (0.7, 0.2, 0.1)
        w_B::NTuple{3, Float64} = (0.65, 0.3, 0.05)
        budgets::NTuple{2, Union{Nothing, Float64}} = (0.08 * T, 0.04 * T)
        floor::Union{Nothing, Float64} = nothing
    end

    function algorithm_options(; kwargs...)
        defaults = (; shooting = :single, nlp_solver = :auto)
        return Options(; merge(defaults, (; kwargs...))...)
    end

    function dynamics!(dx, x, controls, t, a)
        u = controls
        dx[1] = (a.mumax[1] * x[4] / (a.K[1] + x[4]) * (1 + u[1]) / (1 + a.alpha[1] * x[3]) - a.d[1]) * x[1]
        dx[2] = (a.mumax[2] * x[4] / (a.K[2] + x[4]) * (1 + u[2]) / (1 + a.alpha[2] * x[3]) - a.d[2]) * x[2]
        dx[3] = a.g[1] * u[1] * x[1] + a.g[2] * u[2] * x[2] - a.dg * x[3]
        dx[4] = a.D * (a.Sin - x[4]) - sum(a.mumax[j] * x[4] / (a.K[j] + x[4]) * (1 + u[j]) / (1 + a.alpha[j] * x[3]) * x[j] / a.Y[j] for j in 1:2)
        dx[5] = (-a.w_A[1] * x[1] + a.w_A[2] * u[1]^2 + a.w_A[3] * x[3]^2) / a.T
        dx[6] = (-a.w_B[1] * x[2] + a.w_B[2] * u[2]^2 + a.w_B[3] * x[3]^2) / a.T
        dx[7] = u[1]^2
        dx[8] = u[2]^2
        return nothing
    end

    function objective(x, player, a)
        return x[4 + player]
    end

    function constraints(a)
        bounds = [
            StateConstraint(; player = i, index = 6 + i, label = "squared budget u$(subscript(i))", upper = a.budgets[i])
                for i in 1:2 if a.budgets[i] !== nothing
        ]
        a.floor === nothing || append!(bounds, [StateConstraint(; player = i, index = 1, label = "floor x$(subscript(1))", lower = a.floor, path = true) for i in 1:2])
        # Number visible bounds in plot order; shared copies retain one label.
        numbers = Dict{Tuple{Int, Float64, Float64, Bool}, Int}()
        return map(bounds) do c
            startswith(c.label, "!") && return c
            key = (c.index, c.lower, c.upper, c.path)
            i = get!(numbers, key, length(numbers) + 1)
            StateConstraint(;
                player = c.player, index = c.index, label = "g$(subscript(i)): $(c.label)",
                lower = c.lower, upper = c.upper, path = c.path
            )
        end
    end

    function validate_parameters(a)
        for b in a.budgets
            b === nothing || (isfinite(b) && b >= 0) || error("Invalid effort budget")
        end
        a.T > 0 || error("T must be positive")
        all(isfinite, a.x0) || error("Initial states must be finite")
        a.floor === nothing || (isfinite(a.floor) && 0 <= a.floor <= a.x0[1]) ||
            error("Optional floor must be nonnegative and initially feasible")
        all(isfinite, a.initial) && all(0 .<= a.initial .<= 1) || error("Initial controls must lie in [0,1]")
        for w in (a.w_A, a.w_B)
            all(isfinite, w) && all(>=(0), w) && isapprox(sum(w), 1.0; atol = 1.0e-12) ||
                error("Objective weights must be nonnegative and sum to one")
        end
        return
    end

    function game(a = Parameters())
        validate_parameters(a)
        return Game(;
            name = "GenomeSize", state_labels = ["B₁: lineage A biomass", "B₂: lineage B biomass", "G: expression burden", "S: substrate"], control_labels = ["u₁: gene expression", "u₂: gene expression"], owners = [1, 2],
            dynamics! = (dx, x, u, t) -> dynamics!(dx, x, u, t, a), objective = (x, i) -> objective(x, i, a),
            constraints = constraints(a),
            x0 = vcat(collect(a.x0), zeros(4)), T = a.T, quadratures = collect(5:8),
            lower = zeros(2), upper = ones(2), initial = collect(a.initial),
            running_objectives = collect(5:6), metadata = a
        )
    end

    function main(;
            shooting = :single, parameters = Parameters(), options = algorithm_options(; shooting),
            folder = joinpath(@__DIR__, options.shooting == :single ? "output" : "output_multiple"), synthetic = false
        )
        g = game(parameters)
        synthetic && return create_synthetic_data(g, folder)
        return run_game(g; options, folder)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    CorleoneGame.run_logged(GenomeSize.main, @__DIR__)
end
