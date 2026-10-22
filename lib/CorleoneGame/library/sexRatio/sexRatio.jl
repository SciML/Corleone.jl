# Category: seasonal, population, competition, variablePlayers
# Foundress allocation within one brood precedes mating and dispersal.
# Synthetic literature-motivated game.
using CorleoneGame
module SexRatio
    using CorleoneGame

    Base.@kwdef struct Parameters
        N::Int = 3
        T::Float64 = 6.0
        b::Vector{Float64} = [1.0 / (1 + 0.15 * (i - 1)) for i in 1:N]
        df::Float64 = 0.12
        dm::Float64 = 0.2
        supply::Float64 = 0.5
        seasonality::Float64 = 0.8
        male_cost::Float64 = 2.0
        decay::Float64 = 0.1
        cost::Float64 = 0.15
        K::Float64 = 1.0
        mate_half::Float64 = 0.5
        x0::Vector{Float64} = vcat(2.0, fill(0.1, N), fill(0.1, N))
        w_A::NTuple{3, Float64} = (0.8, 0.1, 0.1)
        w_B::NTuple{3, Float64} = (0.65, 0.2, 0.15)
        w_C::NTuple{3, Float64} = (0.7, 0.05, 0.25)
        extra_weights::Matrix{Float64} = repeat(reshape(collect(w_C), 1, 3), max(N - 3, 0), 1)
        budgets::Vector{Union{Nothing, Float64}} = [T * (0.08 + 0.05 * mod(i - 1, 3)) for i in 1:N]
        floor::Union{Nothing, Float64} = 1.5
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
        R = x[1]
        M = sum(x[1 + n + i] for i in 1:n)
        F = sum(x[1 + i] for i in 1:n)
        supply = a.supply * (1 - a.seasonality * cos(2pi * t / a.T))
        # Male-biased provisioning is more resource-intensive in this scenario.
        dx[1] = supply - a.decay * R - a.cost * sum(a.b[i] * (1 + a.male_cost * u[i]) for i in 1:n) * R / (a.K + R)
        for i in 1:n
            births = a.b[i] * R / (a.K + R)
            dx[1 + i] = births * (1 - u[i]) - a.df * x[1 + i]
            dx[1 + n + i] = births * u[i] - a.dm * x[1 + n + i]
            w = weights(a, i)
            # Female success and male mating share both vanish without males.
            success = (x[1 + i] * M + x[1 + n + i] * F) / (a.mate_half + M)
            dx[q + i] = (-w[1] * success + w[2] * (u[i] - 0.5)^2 + w[3] * u[i]^2) / a.T
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
        end
        for i in 1:a.N
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
        size(a.extra_weights) == (max(a.N - 3, 0), 3) || error("Invalid extra_weights dimensions")
        for f in fieldnames(Parameters)
            f in (:floor, :budgets) && continue
            v = getfield(a, f)
            all(isfinite, v isa Number ? (v,) : v) || error("Nonfinite parameter $f")
            all(>=(0), v isa Number ? (v,) : v) || error("Negative parameter $f")
            v isa Vector && f != :x0 && length(v) != a.N && error("Wrong length for $f")
        end
        0 <= a.seasonality <= 1 || error("Seasonality must lie in [0,1]")
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
            name = "SexRatio", state_labels = vcat(["R: brood resource"], ["F$(subscript(i)): daughters" for i in 1:a.N], ["M$(subscript(i)): sons" for i in 1:a.N]),
            control_labels = ["u$(subscript(i)): male allocation fraction" for i in 1:a.N], owners = collect(1:a.N),
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
    CorleoneGame.run_logged(SexRatio.main, @__DIR__)
end
