# Category: episodical
# Slow remodeling follows changed flow demand; this is not a heartbeat cycle.
# Vascular dynamic-game case specification.
using CorleoneGame
module Vascular
    using CorleoneGame

    Base.@kwdef struct Parameters
        initial::NTuple{2, Float64} = (0.2, 0.3)
        T::Float64 = 5.0
        dq::Float64 = 0.4
        R0::NTuple{2, Float64} = (2.0, 2.4)
        target::NTuple{2, Float64} = (0.5, 0.8)
        gamma::NTuple{2, Float64} = (0.5, 0.4)
        kappa::Float64 = 0.8
        P0::Float64 = 1.0
        alpha::Float64 = 0.3
        dv::Float64 = 0.1
        x0::NTuple{6, Float64} = (0.2, 0.2, 2, 2.4, 1, 0)
        w_A::NTuple{3, Float64} = (0.5, 0.2, 0.3)
        w_B::NTuple{3, Float64} = (0.1, 0.5, 0.4)
        budgets::NTuple{2, Union{Nothing, Float64}} = (0.04 * T, 0.03 * T)
        floor::Union{Nothing, Float64} = nothing
    end

    function algorithm_options(; kwargs...)
        defaults = (; shooting = :single, nlp_solver = :auto)
        return Options(; merge(defaults, (; kwargs...))...)
    end

    function dynamics!(dx, x, controls, t, a)
        u = controls
        dx[1] = u[1] - a.dq * x[1]
        dx[2] = u[2] - a.dq * x[2]
        dx[3] = a.dq * (a.R0[1] - x[3]) - a.gamma[1] * u[1] * x[3]
        dx[4] = a.dq * (a.R0[2] - x[4]) - a.gamma[2] * u[2] * x[4]
        dx[5] = a.kappa * (a.P0 - (1 + x[1] + x[2]) * x[5])
        dx[6] = a.alpha * (x[1] + x[2]) - a.dv * x[6]
        dx[7] = (a.w_A[1] * (x[1] - a.target[1])^2 + a.w_A[2] * x[6]^2 + a.w_A[3] * (x[5] - 1)^2) / a.T
        dx[8] = (a.w_B[1] * (x[2] - a.target[2])^2 + a.w_B[2] * x[6]^2 + a.w_B[3] * (x[5] - 1)^2) / a.T
        dx[9] = u[1]^2
        dx[10] = u[2]^2
        return nothing
    end

    function objective(x, player, a)
        return x[6 + player]
    end

    function constraints(a)
        bounds = [
            StateConstraint(; player = i, index = 8 + i, label = "squared budget u$(subscript(i))", upper = a.budgets[i])
                for i in 1:2 if a.budgets[i] !== nothing
        ]
        a.floor === nothing || append!(bounds, [StateConstraint(; player = i, index = 5, label = "floor x$(subscript(5))", lower = a.floor, path = true) for i in 1:2])
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
        a.floor === nothing || (isfinite(a.floor) && 0 <= a.floor <= a.x0[5]) ||
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
            name = "Vascular", state_labels = ["Q₁: branch flow A", "Q₂: branch flow B", "R₁: resistance A", "R₂: resistance B", "P: perfusion pressure", "V: vascular investment"], control_labels = ["u₁: vascular remodeling", "u₂: vascular remodeling"], owners = [1, 2],
            dynamics! = (dx, x, u, t) -> dynamics!(dx, x, u, t, a), objective = (x, i) -> objective(x, i, a),
            constraints = constraints(a),
            x0 = vcat(collect(a.x0), zeros(4)), T = a.T, quadratures = collect(7:10),
            lower = zeros(2), upper = ones(2), initial = collect(a.initial),
            running_objectives = collect(7:8), metadata = a
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
    CorleoneGame.run_logged(Vascular.main, @__DIR__)
end
