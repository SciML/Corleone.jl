# Category: episodical, population, competition, variablePlayers
# One foraging bout changes reserves and retained harvest.
# Synthetic literature-motivated game.
using CorleoneGame
module Foraging
    using CorleoneGame

    Base.@kwdef struct Parameters
        N::Int = 3
        T::Float64 = 10.0
        r::Float64 = 0.8
        K::Float64 = 10.0
        c::Vector{Float64} = [0.12 + 0.04 * (i - 1) for i in 1:N]
        e::Vector{Float64} = [0.8 / (1 + 0.1 * (i - 1)) for i in 1:N]
        h::Vector{Float64} = [0.6 / (1 + 0.1 * (i - 1)) for i in 1:N]
        de::Float64 = 0.3
        dh::Float64 = 0.1
        x0::Vector{Float64} = vcat(8.0, ones(N), zeros(N))
        # Weights (retained harvest, squared effort); no preference for the patch level.
        w_A::NTuple{2, Float64} = (1 / 3, 2 / 3)
        w_B::NTuple{2, Float64} = (0.375, 0.625)
        w_C::NTuple{2, Float64} = (0.8, 0.2)
        extra_weights::Matrix{Float64} = repeat(reshape(collect(w_C), 1, 2), max(N - 3, 0), 1)
        budgets::Vector{Union{Nothing, Float64}} = [T * (0.35 + 0.05 * mod(i - 1, 3)) for i in 1:N]
        floor::Union{Nothing, Float64} = 7.0
        initial::Vector{Float64} = fill(0.2, N)
    end

    weights(a, i) = i == 1 ? a.w_A : i == 2 ? a.w_B : i == 3 ? a.w_C : Tuple(a.extra_weights[i - 3, :])

    function algorithm_options(; kwargs...)
        defaults = (; shooting = :single, nlp_solver = :ipopt, warm_start = true, ode_tolerance = 1.0e-10)
        return Options(; merge(defaults, (; kwargs...))...)
    end

    function dynamics!(dx, x, u, t, a)
        n = a.N
        q = 1 + 2n
        S = x[1]
        dx[1] = a.r * S * (1 - S / a.K) - S * sum(a.c[i] * u[i] for i in 1:n)
        for i in 1:n
            dx[1 + i] = a.e[i] * u[i] * S - a.de * x[1 + i]
            dx[1 + n + i] = a.h[i] * u[i] * x[1 + i] - a.dh * x[1 + n + i]
            w = weights(a, i)
            dx[q + i] = (-w[1] * x[1 + n + i] + w[2] * u[i]^2) / a.T
            dx[q + n + i] = u[i]^2
        end
        return nothing
    end

    objective(x, i, a) = x[1 + 2a.N + i]

    function constraints(a)
        bounds = StateConstraint[]
        for i in 1:a.N
            a.budgets[i] === nothing || push!(
                bounds, StateConstraint(;
                    player = i, index = 1 + 3a.N + i,
                    label = "squared budget u$(subscript(i))", upper = a.budgets[i]
                )
            )
            a.floor === nothing || push!(bounds, StateConstraint(; player = i, index = 1, label = "floor x$(subscript(1))", lower = a.floor, path = true))
        end
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
        a.N >= 2 || error("At least two players required")
        length(a.x0) == 1 + 2a.N && length(a.initial) == length(a.budgets) == a.N || error("State/control dimensions disagree")
        size(a.extra_weights) == (max(a.N - 3, 0), 2) || error("Invalid extra_weights dimensions")
        for f in fieldnames(Parameters)
            f in (:floor, :budgets) && continue
            v = getfield(a, f)
            all(isfinite, v isa Number ? (v,) : v) || error("Nonfinite parameter $f")
            all(>=(0), v isa Number ? (v,) : v) || error("Negative parameter $f")
            v isa Vector && f != :x0 && length(v) != a.N && error("Wrong length for $f")
        end
        a.T > 0 && a.K > 0 || error("Positive horizon and saturation scale required")
        all(0 .<= a.initial .<= 1) || error("Initial controls outside [0,1]")
        a.floor === nothing || (isfinite(a.floor) && 0 <= a.floor <= a.x0[1]) || error("Invalid resource floor")
        for i in 1:a.N
            w = weights(a, i)
            all(>=(0), w) && isapprox(sum(w), 1; atol = 1.0e-12) || error("Weights must sum to one")
            b = a.budgets[i]
            b === nothing || (isfinite(b) && b >= 0) || error("Invalid budget")
        end
        return
    end

    function game(a = Parameters())
        validate_parameters(a)
        q = 1 + 2a.N
        return Game(;
            name = "Foraging", state_labels = vcat(["S: food"], ["E$(subscript(i)): reserve" for i in 1:a.N], ["H$(subscript(i)): retained harvest" for i in 1:a.N]),
            control_labels = ["u$(subscript(i)): foraging effort" for i in 1:a.N], owners = collect(1:a.N),
            dynamics! = (dx, x, u, t) -> dynamics!(dx, x, u, t, a), objective = (x, i) -> objective(x, i, a),
            constraints = constraints(a), x0 = vcat(a.x0, zeros(2a.N)), T = a.T,
            quadratures = collect((q + 1):(q + 2a.N)), running_objectives = collect((q + 1):(q + a.N)),
            lower = zeros(a.N), upper = ones(a.N), initial = copy(a.initial), metadata = a
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
    CorleoneGame.run_logged(Foraging.main, @__DIR__)
end
