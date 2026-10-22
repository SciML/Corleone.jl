# Category: seasonal, population, competition, variablePlayers
# Colony recruitment during a breeding season precedes omitted overwintering.
# Proposed literature-motivated game.
using CorleoneGame
module AntColonies
    using CorleoneGame

    Base.@kwdef struct Parameters
        N::Int = 3
        T::Float64 = 6.0
        food_supply::Float64 = 0.8
        water_supply::Float64 = 0.7
        resource_loss::Float64 = 0.15
        food_rate::Vector{Float64} = [0.8 + 0.1 * i for i in 1:N]
        water_rate::Vector{Float64} = [0.9 - 0.05 * i / N for i in 1:N]
        half::Float64 = 0.4
        # Strong interference and conflict make early territorial aggression worthwhile.
        interference::Float64 = 6.0
        birth::Float64 = 1.2
        maturation::Float64 = 0.6
        brood_loss::Float64 = 0.08
        worker_loss::Float64 = 0.12
        foraging_risk::Float64 = 0.08
        conflict::Float64 = 0.6
        store_loss::Float64 = 0.1
        brood_cost::Float64 = 0.8
        x0::Vector{Float64} = vcat(3.0, 3.0, ones(N), fill(0.3, N), fill(0.3, N))
        weights::Matrix{Float64} = [k == 1 ? 0.15 : k == 2 ? 0.15 + 0.15 * (i - 1) / N : 0.7 - 0.15 * (i - 1) / N for i in 1:N, k in 1:3]
        budgets::Vector{Union{Nothing, Float64}} = fill(0.15 * T, N)
        floors::Vector{Union{Nothing, Float64}} = fill(0.1, N)
        initial::Vector{Float64} = fill(0.2, 3N)
    end

    player_number(a) = a.N
    physical_number(a) = (n = player_number(a); 2 + 3n)
    control_number(a) = (n = player_number(a); 3n)

    function algorithm_options(; kwargs...)
        # Undamped responses converge faster; denser constraint sampling matches the
        # independent feasibility audit of the binding food-store floors.
        defaults = (; shooting = :single, nlp_solver = :ipopt, ode_tolerance = 1.0e-9, damping = 1.0, constraint_samples = 8)
        return Options(; merge(defaults, (; kwargs...))...)
    end

    function dynamics!(dx, x, u, t, a)
        n = player_number(a)
        q = physical_number(a)
        food = zero(x[1]); water = zero(x[1])
        shares = reshape(u, 3, n) ./ (1 .+ sum(reshape(u, 3, n); dims = 1))
        for i in 1:n
            A, B, C = x[2 + i], x[2 + n + i], x[2 + 2n + i]
            rivals = sum(shares[3, j] * x[2 + j] for j in 1:n if j != i)
            harvest = a.food_rate[i] * shares[1, i] * A * x[1] / (a.half + x[1]) / (1 + a.interference * rivals)
            drink = a.water_rate[i] * shares[2, i] * A * x[2] / (a.half + x[2])
            production = a.birth * A * C / (a.half + C) * drink / (a.half + drink)
            dx[2 + i] = a.maturation * B - (a.worker_loss + a.foraging_risk * shares[1, i] + a.conflict * rivals / (1 + shares[3, i] * A)) * A
            dx[2 + n + i] = production - (a.maturation + a.brood_loss) * B
            dx[2 + 2n + i] = harvest - a.store_loss * C - a.brood_cost * production
            food += harvest
            water += drink
            effort = sum(u[(3i - 2):3i] .^ 2)
            features = (-B, effort, -A)
            dx[q + i] = sum(a.weights[i, k] * features[k] for k in 1:3) / a.T
            dx[q + n + i] = effort
        end
        dx[1] = a.food_supply - a.resource_loss * x[1] - food
        dx[2] = a.water_supply - a.resource_loss * x[2] - water
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
                    label = "squared budget " * join(["u$(subscript(j))" for j in (3i - 2):3i], ", "), upper = a.budgets[i]
                )
            )
        end
        for i in 1:n
            a.floors[i] === nothing || push!(bounds, StateConstraint(; player = i, index = 2 + 2n + i, label = "floor x$(subscript(2 + 2n + i))", lower = a.floors[i], path = true))
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
        a.N >= 2 || error("At least two colonies required")
        length(a.food_rate) == length(a.water_rate) == n || error("Colony rate dimensions disagree")
        a.half > 0 || error("Positive saturation scale required")
        length(a.x0) == q && length(a.initial) == control_number(a) || error("State/control dimensions disagree")
        length(a.budgets) == n && size(a.weights) == (n, 3) || error("Budget or weight dimensions disagree")
        indices = collect((3 + 2n):(2 + 3n))
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
            name = "AntColonies", state_labels = vcat(["F: food", "W: water"], ["A$(subscript(i)): workers" for i in 1:n], ["B$(subscript(i)): brood" for i in 1:n], ["C$(subscript(i)): food store" for i in 1:n]),
            control_labels = ["u$(subscript(3i - 3 + k)): $meaning" for i in 1:n for (k, meaning) in enumerate(("food foraging", "water foraging", "aggression"))], owners = repeat(collect(1:n); inner = 3),
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
    CorleoneGame.run_logged(AntColonies.main, @__DIR__)
end
