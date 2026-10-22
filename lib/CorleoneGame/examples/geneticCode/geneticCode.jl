# Category: episodical
# One expression-demand episode reallocates proofreading and substrate pools.
# GeneticCode dynamic-game case specification.
using CorleoneGame
module GeneticCode
    using CorleoneGame

    Base.@kwdef struct Parameters
        initial::NTuple{2, Float64} = (0.2, 0.3)
        T::Float64 = 10.0
        s::NTuple{2, Float64} = (1.0, 0.9)
        d::NTuple{2, Float64} = (0.4, 0.45)
        q::NTuple{2, Float64} = (0.2, 0.25)
        rho::NTuple{2, Float64} = (0.3, 0.2)
        qt::Float64 = 0.4
        dt::Float64 = 0.2
        qr::Float64 = 0.3
        dr::Float64 = 0.25
        x0::NTuple{4, Float64} = (1.0, 1.0, 0.0, 0.0)
        w_A::NTuple{2, Float64} = (0.3, 0.7)
        w_B::NTuple{2, Float64} = (0.8, 0.2)
        budgets::NTuple{2, Union{Nothing, Float64}} = (0.01 * T, 0.03 * T)
        floor::Union{Nothing, Float64} = nothing
    end

    function algorithm_options(; kwargs...)
        defaults = (; shooting = :single, nlp_solver = :auto)
        return Options(; merge(defaults, (; kwargs...))...)
    end

    function dynamics!(dx, x, controls, t, a)
        u = controls
        dx[1] = a.s[1] - a.d[1] * x[1] - a.q[1] * x[1] - a.rho[1] * u[1] * x[1] / (1 + u[2])
        dx[2] = a.s[2] - a.d[2] * x[2] - a.q[2] * x[2] - a.rho[2] * u[2] * x[2] / (1 + u[1])
        dx[3] = a.qt * x[1] - a.dt * x[3]
        dx[4] = a.qr * x[2] - a.dr * x[4] + a.rho[1] * u[1] * x[1] / (1 + u[2]) + a.rho[2] * u[2] * x[2] / (1 + u[1])
        dx[5] = (a.w_A[1] * (x[1] - 1)^2 + a.w_A[2] * u[1]^2) / a.T
        dx[6] = (a.w_B[1] * (x[2] - 1)^2 + a.w_B[2] * u[2]^2) / a.T
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
            name = "GeneticCode", state_labels = ["C₁: codon pool A", "C₂: codon pool B", "Q: throughput proxy", "R: processed-product proxy"], control_labels = ["u₁: proofreading", "u₂: proofreading"], owners = [1, 2],
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
    CorleoneGame.run_logged(GeneticCode.main, @__DIR__)
end
