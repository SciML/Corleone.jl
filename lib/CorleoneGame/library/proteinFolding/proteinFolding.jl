# Category: episodical, molecular, regulation
# Processing a finite protein pool requires external replenishment for another episode.
# ProteinFolding dynamic-game case specification.
using CorleoneGame
module ProteinFolding
    using CorleoneGame

    Base.@kwdef struct Parameters
        initial::NTuple{2, Float64} = (0.2, 0.3)
        T::Float64 = 8.0
        kf::Float64 = 1.5
        ku::Float64 = 0.6
        kn::Float64 = 0.3
        du::Float64 = 0.02
        dn::Float64 = 0.01
        c::NTuple{2, Float64} = (0.8, 0.5)
        # Chaperone capacity is consumed by assisted folding and replenished slowly.
        ca::Float64 = 2.0
        da::Float64 = 0.2
        A0::Float64 = 0.2
        x0::NTuple{4, Float64} = (0.8, 0.2, 0.0, 1.0)
        w_A::NTuple{2, Float64} = (0.85, 0.15)
        w_B::NTuple{2, Float64} = (0.75, 0.25)
        budgets::NTuple{2, Union{Nothing, Float64}} = (0.05 * T, 0.03 * T)
        floor::Union{Nothing, Float64} = nothing
    end

    function algorithm_options(; kwargs...)
        defaults = (; shooting = :single, nlp_solver = :auto)
        return Options(; merge(defaults, (; kwargs...))...)
    end

    function dynamics!(dx, x, controls, t, a)
        u = controls
        dx[1] = -a.kf * x[1] + a.ku * x[2] - a.du * x[1]
        dx[2] = a.kf * x[1] - (a.ku + a.kn) * x[2] - (a.c[1] * u[1] + a.c[2] * u[2]) * x[4] * x[2]
        dx[3] = a.kn * x[2] - a.dn * x[3] + (a.c[1] * u[1] + a.c[2] * u[2]) * x[4] * x[2]
        dx[4] = a.A0 - a.ca * (u[1] + u[2]) * x[4] * x[2] - a.da * x[4]
        dx[5] = (a.w_A[1] * (x[3] - 1)^2 + a.w_A[2] * u[1]^2) / a.T
        dx[6] = (a.w_B[1] * (x[3] - 1)^2 + a.w_B[2] * u[2]^2) / a.T
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
        a.floor === nothing || append!(bounds, [StateConstraint(; player = i, index = 3, label = "floor x$(subscript(3))", lower = a.floor, path = true) for i in 1:2])
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
        a.floor === nothing || (isfinite(a.floor) && 0 <= a.floor <= a.x0[3]) ||
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
            name = "ProteinFolding", state_labels = ["U: unfolded protein", "I: folding intermediate", "N: native protein", "Ch: chaperone capacity"], control_labels = ["u₁: chaperone effort", "u₂: chaperone effort"], owners = [1, 2],
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
    CorleoneGame.run_logged(ProteinFolding.main, @__DIR__)
end
