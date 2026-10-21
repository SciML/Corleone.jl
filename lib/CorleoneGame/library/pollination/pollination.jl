# Category: seasonal, population, mutualism, variablePlayers
# A flowering season precedes reproduction, dispersal, and other life-history phases.
# Proposed literature-motivated game.
using CorleoneGame
module Pollination
    using CorleoneGame

    Base.@kwdef struct Parameters
        plants::Int = 2
        pollinators::Int = 2
        T::Float64 = 8.0
        r::Float64 = 0.5
        K::Float64 = 2.0
        benefit::Float64 = 0.4
        nectar_cost::Float64 = 0.04
        nectar_rate::Float64 = 1.5
        nectar_loss::Float64 = 0.5
        half::Float64 = 0.3
        visit::Matrix{Float64} = [(i == mod1(j, plants) ? 0.95 : 0.2) * (1 + 0.15 * (j - 1)) for j in 1:pollinators, i in 1:plants]
        conversion::Float64 = 1.5
        mortality::Float64 = 0.2
        crowding::Float64 = 0.06
        effort_cost::Float64 = 0.04
        x0::Vector{Float64} = vcat(
            [0.6 + 0.9 * (i - 1) for i in 1:plants], [0.1 + 0.7 * (i - 1) for i in 1:plants],
            [0.25 + 0.5 * (j - 1) for j in 1:pollinators]
        )
        weights::Matrix{Float64} = [
            i <= plants ? (isodd(i) ? (0.88, 0.08, 0.04)[k] : (0.65, 0.3, 0.05)[k]) :
                (isodd(i - plants) ? (0.8, 0.06, 0.14)[k] : (0.5, 0.3, 0.2)[k]) for i in 1:(plants + pollinators), k in 1:3
        ]
        budgets::Vector{Union{Nothing, Float64}} = [T * (i <= plants ? (isodd(i) ? 0.12 : 0.04) : (isodd(i - plants) ? 0.18 : 0.08)) for i in 1:(plants + pollinators)]
        floors::Vector{Union{Nothing, Float64}} = [i <= plants ? 0.12 : 0.1 for i in 1:(plants + pollinators)]
        #    budgets::Vector{Union{Nothing,Float64}} = fill(0.6*T,plants+pollinators)
        #    floors::Vector{Union{Nothing,Float64}} = fill(0.1,plants+pollinators)
        initial::Vector{Float64} = fill(0.2, plants + plants * pollinators)
    end

    player_number(a) = a.plants + a.pollinators
    physical_number(a) = (n = player_number(a); 2a.plants + a.pollinators)
    control_number(a) = (n = player_number(a); a.plants + a.plants * a.pollinators)

    function algorithm_options(; kwargs...)
        defaults = (; shooting = :single, nlp_solver = :ipopt, ode_tolerance = 1.0e-9)
        return Options(; merge(defaults, (; kwargs...))...)
    end

    function dynamics!(dx, x, u, t, a)
        n = player_number(a)
        q = physical_number(a)
        p = a.plants
        visits = [a.visit[j, i] * u[p + (j - 1) * p + i] / (1 + sum(u[(p + (j - 1) * p + 1):(p + j * p)])) * x[2p + j] * x[p + i] / (a.half + x[p + i]) for j in 1:a.pollinators, i in 1:p]
        for i in 1:p
            received = sum(visits[:, i])
            dx[i] = x[i] * (a.r * (1 - x[i] / a.K) + a.benefit * received / (a.half + x[i]) - a.nectar_cost * u[i]^2)
            dx[p + i] = a.nectar_rate * u[i] * x[i] - a.nectar_loss * x[p + i] - received
            features = (-x[i], u[i]^2, x[p + i]^2)
            dx[q + i] = sum(a.weights[i, k] * features[k] for k in 1:3) / a.T
            dx[q + n + i] = u[i]^2
        end
        for j in 1:a.pollinators
            i = p + j
            intake = sum(visits[j, :])
            effort = sum(u[(p + (j - 1) * p + 1):(p + j * p)] .^ 2)
            A = x[2p + j]
            dx[2p + j] = a.conversion * intake - (a.mortality + a.crowding * A + a.effort_cost * effort) * A
            features = (-A, effort, 1 / (1 + intake))
            dx[q + i] = sum(a.weights[i, k] * features[k] for k in 1:3) / a.T
            dx[q + n + i] = effort
        end
        return nothing
    end

    objective(x, i, a) = x[physical_number(a) + i]

    function constraints(a)
        n = player_number(a); q = physical_number(a)
        bounds = StateConstraint[]
        for i in 1:n
            a.budgets[i] === nothing || push!(
                bounds, StateConstraint(;
                    player = i, index = q + n + i,
                    label = "squared budget " * join(["u$(subscript(j))" for j in (i <= a.plants ? (i:i) : ((a.plants + (i - a.plants - 1) * a.plants + 1):(a.plants + (i - a.plants) * a.plants)))], ", "), upper = a.budgets[i]
                )
            )
        end
        indices = vcat(collect(1:a.plants), collect((2a.plants + 1):q))
        for i in 1:n
            a.floors[i] === nothing || push!(bounds, StateConstraint(; player = i, index = indices[i], label = "floor x$(subscript(indices[i]))", lower = a.floors[i], path = true))
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
        n = player_number(a); q = physical_number(a)
        a.plants >= 2 && a.pollinators >= 2 || error("At least two plants and two pollinators required")
        size(a.visit) == (a.pollinators, a.plants) && a.K > 0 && a.half > 0 || error("Invalid visitation matrix or saturation scales")
        length(a.x0) == q && length(a.initial) == control_number(a) || error("State/control dimensions disagree")
        length(a.budgets) == n && size(a.weights) == (n, 3) || error("Budget or weight dimensions disagree")
        indices = vcat(collect(1:a.plants), collect((2a.plants + 1):q))
        length(a.floors) == length(indices) || error("Floor dimensions disagree")
        for name in fieldnames(Parameters)
            name in (:budgets, :floors) && continue
            value = getfield(a, name)
            values = value isa Number ? (value,) : value
            all(isfinite, values) && all(>=(0), values) || error("Parameters must be finite and nonnegative: $name")
        end
        a.T > 0 && all(a.initial .<= 1) || error("Invalid horizon or initial strategy")
        all(isapprox.(vec(sum(a.weights; dims = 2)), 1; atol = 1.0e-12)) || error("Each weight row must sum to one")
        for b in a.budgets
            b === nothing || (isfinite(b) && b >= 0) || error("Invalid budget")
        end
        for (b, k) in zip(a.floors, indices)
            b === nothing || (isfinite(b) && b >= 0) || error("Terminal floor must be finite and nonnegative")
        end
        return
    end

    function game(a = Parameters())
        validate_parameters(a)
        n = player_number(a); q = physical_number(a); nc = control_number(a)
        return Game(;
            name = "Pollination", state_labels = vcat(["P$(subscript(i)): reproductive biomass" for i in 1:a.plants], ["ν$(subscript(i)): nectar" for i in 1:a.plants], ["A$(subscript(j)): pollinators" for j in 1:a.pollinators]),
            control_labels = vcat(["u$(subscript(i)): nectar investment" for i in 1:a.plants], ["u$(subscript(a.plants + (j - 1) * a.plants + i)): plant $i visitation" for j in 1:a.pollinators for i in 1:a.plants]), owners = vcat(collect(1:a.plants), repeat(collect((a.plants + 1):n); inner = a.plants)),
            dynamics! = (dx, x, u, t) -> dynamics!(dx, x, u, t, a), objective = (x, i) -> objective(x, i, a),
            constraints = constraints(a), x0 = vcat(a.x0, zeros(2n)), T = a.T, quadratures = collect((q + 1):(q + 2n)),
            running_objectives = collect((q + 1):(q + n)), lower = zeros(nc), upper = ones(nc), initial = copy(a.initial), metadata = a
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
    CorleoneGame.run_logged(Pollination.main, @__DIR__)
end
