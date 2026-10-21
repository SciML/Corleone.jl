# Category: episodical, cellular, competition, variablePlayers
# One colonization, maturation, and dispersal episode changes the attached community.
# Proposed literature-motivated game.
using CorleoneGame
module Biofilm
    using CorleoneGame

    Base.@kwdef struct Parameters
        N::Int = 3
        T::Float64 = 12.0
        vmax::Vector{Float64} = [1.0 + 0.2 * (i - 1) for i in 1:N]
        half::Float64 = 0.4
        biomass_yield::Float64 = 0.6
        # Low initial EPS and stronger protection reward early matrix investment.
        eps_yield::Float64 = 1.5
        mortality::Float64 = 0.5
        protection::Float64 = 10.0
        crowding::Float64 = 0.08
        eps_loss::Float64 = 0.15
        dilution::Float64 = 0.5
        substrate_in::Float64 = 3.0
        oxygen_in::Float64 = 2.0
        oxygen_cost::Float64 = 0.5
        x0::Vector{Float64} = vcat(
            [0.2 + 0.25 * (i - 1) for i in 1:N],
            [0.01 + 0.025 * (i - 1) + 0.005 * (i - 1)^2 for i in 1:N], 3.0, 2.0
        )
        weights::Matrix{Float64} = [
            k == 1 ? 0.5 - 0.3 * (i - 1) / (N - 1) : k == 2 ? 0.4 + 0.2 * (i - 1) / (N - 1) :
                0.1 + 0.1 * (i - 1) / (N - 1) for i in 1:N, k in 1:3
        ]
        budgets::Vector{Union{Nothing, Float64}} = [0.2 + 0.4 * i for i in 1:N]
        floors::Vector{Union{Nothing, Float64}} = [0.5, 0.4]
        initial::Vector{Float64} = repeat([0.2, 0.1], N)
    end

    player_number(a) = a.N
    physical_number(a) = (n = player_number(a); 2n + 2)
    control_number(a) = (n = player_number(a); 2n)

    function algorithm_options(; kwargs...)
        defaults = (; shooting = :single, nlp_solver = :ipopt, ode_tolerance = 1.0e-9)
        return Options(; merge(defaults, (; kwargs...))...)
    end

    function dynamics!(dx, x, u, t, a)
        n = player_number(a)
        q = physical_number(a)
        S, O = x[2n + 1], x[2n + 2]
        total = zero(S)
        biomass = sum(x[1:n])
        for i in 1:n
            e, d = u[2i - 1], u[2i]
            uptake = a.vmax[i] * S / (a.half + S) * O / (a.half + O) * x[i]
            dx[i] = (1 - e) * a.biomass_yield * uptake - (a.mortality / (1 + a.protection * x[n + i]) + d + a.crowding * biomass) * x[i]
            dx[n + i] = a.eps_yield * e * uptake - a.eps_loss * x[n + i]
            total += uptake
            effort = e^2 + d^2
            features = (-x[i], -d * x[i], effort)
            dx[q + i] = sum(a.weights[i, k] * features[k] for k in 1:3) / a.T
            dx[q + n + i] = effort
        end
        dx[2n + 1] = a.dilution * (a.substrate_in - S) - total
        dx[2n + 2] = a.dilution * (a.oxygen_in - O) - a.oxygen_cost * total
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
                    label = "squared budget u$(subscript(2i - 1)), u$(subscript(2i))", upper = a.budgets[i]
                )
            )
        end
        for i in 1:n, k in 1:2
            a.floors[k] === nothing || push!(bounds, StateConstraint(; player = i, index = 2n + k, label = "floor x$(subscript(2n + k))", lower = a.floors[k], path = true))
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
        a.N >= 2 && length(a.vmax) == n || error("Invalid strain count or uptake rates")
        a.half > 0 || error("Positive saturation scale required")
        length(a.x0) == q && length(a.initial) == control_number(a) || error("State/control dimensions disagree")
        length(a.budgets) == n && size(a.weights) == (n, 3) || error("Budget or weight dimensions disagree")
        indices = collect((2n + 1):(2n + 2))
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
            name = "Biofilm", state_labels = vcat(["B$(subscript(i)): attached biomass" for i in 1:n], ["E$(subscript(i)): local EPS" for i in 1:n], ["S: substrate", "O: oxygen"]),
            control_labels = ["u$(subscript(2i - 2 + k)): $meaning" for i in 1:n for (k, meaning) in enumerate(("EPS investment", "dispersal"))], owners = repeat(collect(1:n); inner = 2),
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
    CorleoneGame.run_logged(Biofilm.main, @__DIR__)
end
