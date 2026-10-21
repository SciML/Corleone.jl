# Category: seasonal, population, competition, variablePlayers
# Root investment and biomass growth occur within a finite growing season.
# Proposed literature-motivated game.
using CorleoneGame
module PlantWater
    using CorleoneGame

    Base.@kwdef struct Parameters
        N::Int = 4
        layers::Int = 3
        T::Float64 = 8.0
        rain::Vector{Float64} = [0.25 + 0.15 * (k - 1) for k in 1:layers]
        seasonality::Float64 = 0.8
        evaporation::Float64 = 0.12
        # Each plant specializes in one layer; deeper layers replenish faster.
        uptake::Matrix{Float64} = [(k == mod1(i, layers) ? 0.9 : 0.3) * (1 + 0.08 * (i - 1)) for i in 1:N, k in 1:layers]
        half::Float64 = 0.4
        root_growth::Float64 = 0.9
        root_loss::Float64 = 0.18
        assimilation::Float64 = 1.2
        mortality::Float64 = 0.05
        crowding::Float64 = 0.03
        root_cost::Float64 = 0.02
        x0::Vector{Float64} = vcat(
            [k == 1 ? 1.8 : k == 2 ? 0.9 : 0.5 for k in 1:layers],
            [0.06 + 0.03 * mod(i + k, 3) for i in 1:N for k in 1:layers], [0.6 + 0.2 * (i - 1) for i in 1:N]
        )
        #    weights::Matrix{Float64} = [k==1 ? 0.85-0.15*(i-1)/(N-1) : k==2 ? 0.0+0.08*(i-1)/(N-1) :
        #        0.15+0.07*(i-1)/(N-1) for i in 1:N,k in 1:3]
        budgets::Vector{Union{Nothing, Float64}} = [1.9 + 0.4 * i for i in 1:N]
        weights::Matrix{Float64} = [
            k == 1 ? 0.85 - 0.15 * (i - 1) / (N - 1) : k == 2 ? 0.06 + 0.08 * (i - 1) / (N - 1) :
                0.09 + 0.07 * (i - 1) / (N - 1) for i in 1:N, k in 1:3
        ]
        #    budgets::Vector{Union{Nothing,Float64}} = [2.4+0.4*i for i in 1:N]
        floors::Vector{Union{Nothing, Float64}} = fill(nothing, layers)
        initial::Vector{Float64} = fill(0.2, N * layers)
    end

    player_number(a) = a.N
    physical_number(a) = (n = player_number(a); a.layers + n * a.layers + n)
    control_number(a) = (n = player_number(a); n * a.layers)

    function algorithm_options(; kwargs...)
        defaults = (; shooting = :single, nlp_solver = :ipopt, ode_tolerance = 1.0e-9)
        return Options(; merge(defaults, (; kwargs...))...)
    end

    function dynamics!(dx, x, u, t, a)
        n = player_number(a)
        q = physical_number(a)
        L = a.layers
        for k in 1:L
            dx[k] = a.rain[k] * (1 + a.seasonality * sin(2pi * t / a.T)) - a.evaporation * x[k]
        end
        for i in 1:n
            B = x[L + n * L + i]
            total = zero(B); roots = zero(B); effort = zero(B)
            for k in 1:L
                j = (i - 1) * L + k
                R = x[L + j]
                uptake = a.uptake[i, k] * R * x[k] / (a.half + x[k])
                dx[k] -= uptake
                dx[L + j] = a.root_growth * u[j] * B / (1 + B) - a.root_loss * R
                total += uptake; roots += R^2; effort += u[j]^2
            end
            dx[L + n * L + i] = B * (a.assimilation * total / (1 + B) - a.mortality - a.crowding * B - a.root_cost * effort)
            features = (-B, effort, roots)
            dx[q + i] = sum(a.weights[i, k] * features[k] for k in 1:3) / a.T
            dx[q + n + i] = total # Water quota, not an effort budget.
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
                    label = "water uptake budget", upper = a.budgets[i]
                )
            )
        end
        for i in 1:n, k in 1:a.layers
            a.floors[k] === nothing || push!(bounds, StateConstraint(; player = i, index = k, label = "floor x$(subscript(k))", lower = a.floors[k], path = true))
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
        a.N >= 2 && a.layers >= 2 || error("At least two plants and soil layers required")
        length(a.rain) == a.layers && size(a.uptake) == (n, a.layers) || error("Water parameter dimensions disagree")
        a.half > 0 && 0 <= a.seasonality <= 1 || error("Invalid saturation scale or seasonal forcing")
        length(a.x0) == q && length(a.initial) == control_number(a) || error("State/control dimensions disagree")
        length(a.budgets) == n && size(a.weights) == (n, 3) || error("Budget or weight dimensions disagree")
        indices = collect(1:a.layers)
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
            b === nothing || (isfinite(b) && 0 <= b <= a.x0[k]) || error("Floor must be nonnegative and initially feasible")
        end
        return
    end

    function game(a = Parameters())
        validate_parameters(a)
        n = player_number(a); q = physical_number(a); nc = control_number(a)
        return Game(;
            name = "PlantWater", state_labels = vcat(["W$(subscript(k)): soil water" for k in 1:a.layers], ["R$(subscript(i)),$(subscript(k)): roots" for i in 1:n for k in 1:a.layers], ["B$(subscript(i)): biomass" for i in 1:n]),
            control_labels = ["u$(subscript((i - 1) * a.layers + k)): layer $k root investment" for i in 1:n for k in 1:a.layers], owners = repeat(collect(1:n); inner = a.layers),
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
    CorleoneGame.run_logged(PlantWater.main, @__DIR__)
end
