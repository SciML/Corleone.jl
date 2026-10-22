# Category: periodical, population, antagonism, variablePlayers
# An intrinsic predator-prey cycle closes every population; its duration belongs to A.
# Proposed literature-motivated game.
using CorleoneGame
module PredatorPrey
    using CorleoneGame

    # Autonomous community: saturating (Holling type II) predation on enriched prey
    # produces a stable limit cycle under the constant reference strategies
    # (paradox of enrichment), so no external clock is needed and the cycle duration
    # is a decision of player A (prey 1), as in circadianCycle. Dynamics are written
    # on normalized time s in [0,1] and multiplied by the duration.
    # Default x0, duration, and scale come from a reference-cycle preparation:
    # x0 is the phase point where prey 1 crosses its cycle mean upwards; scale holds the
    # reference cycle means. Custom kinetics or sizes require a new cycle preparation.
    # Approximate closure is not a claim of long-run orbital stability or equilibrium.
    Base.@kwdef struct Parameters
        prey::Int = 3
        predators::Int = 2
        # Absolute return band. With 1e-4 or 1e-3 all shared return conditions are active for
        # every player and sequential best responses creep along a continuum of generalized
        # equilibria without converging within the round limits (tested 2026-09-24).
        periodicity_tolerance::Float64 = 1.0e-2
        r::Vector{Float64} = [0.8 + 0.1 * i for i in 1:prey]
        K::Vector{Float64} = fill(3.0, prey)
        attack::Matrix{Float64} = [3 * (0.18 + 0.04 * mod(i + j, 3)) for j in 1:predators, i in 1:prey]
        handling::Float64 = 0.5
        conversion::Float64 = 0.45
        mortality::Vector{Float64} = [0.25 + 0.05 * j for j in 1:predators]
        crowding::Float64 = 0.03
        growth_benefit::Float64 = 0.25
        strategy_cost::Float64 = 0.12
        default_size::Bool = prey == 3 && predators == 2
        x0::Vector{Float64} = default_size ? [0.19811849641950602, 1.430316130098483, 1.0998866937687062, 0.8595951089451239, 0.4409164491899876] :
            vcat(fill(1.0, prey), fill(0.7, predators)) # Custom sizes require cycle preparation.
        scale::Vector{Float64} = default_size ? [0.1988327165668489, 0.9306449028547795, 0.7068407768239765, 1.259519776729533, 0.6566602541452006] : copy(x0)
        # A owns the cycle duration; the reference limit cycle has period `duration`.
        duration::Float64 = default_size ? 16.582148321859844 : 12.0
        duration_bounds::NTuple{2, Float64} = (0.5 * duration, 2.0 * duration)
        weights::Matrix{Float64} = [i <= prey ? (k == 1 ? 0.65 : k == 2 ? 0.25 : 0.1) : (k == 1 ? 0.75 : k == 2 ? 0.2 : 0.05) for i in 1:(prey + predators), k in 1:3]
        # Mean squared effort per unit time over one cycle, independent of its duration.
        budgets::Vector{Union{Nothing, Float64}} = fill(0.35, prey + predators)
        floors::Vector{Union{Nothing, Float64}} = fill(nothing, prey + predators)
        initial::Vector{Float64} = fill(0.2, 2 * (prey + predators))
    end

    player_number(a) = a.prey + a.predators
    physical_number(a) = (n = player_number(a); n)
    control_number(a) = (n = player_number(a); 2n)

    function algorithm_options(; kwargs...)
        # Undamped responses: averaged profiles violate the shared cycle closure.
        defaults = (; shooting = :single, nlp_solver = :ipopt, ode_tolerance = 1.0e-9, damping = 1.0)
        return Options(; merge(defaults, (; kwargs...))...)
    end

    "Per-capita rates on physical time; u holds all strategy controls."
    function rates!(dx, x, u, a)
        # Effective attack of predator j on prey k, reduced by prey defense and raised by hunting.
        attack(j, k) = a.attack[j, k] * (1 + u[2 * (a.prey + j) - 1]) / (1 + u[2k - 1])
        saturation = [1 + a.handling * sum(attack(j, k) * x[k] for k in 1:a.prey) for j in 1:a.predators]
        for i in 1:a.prey
            d, g = u[2i - 1], u[2i]
            predation = sum(attack(j, i) * x[a.prey + j] / saturation[j] for j in 1:a.predators)
            dx[i] = x[i] * (a.r[i] * (1 - x[i] / a.K[i]) + a.growth_benefit * g - a.strategy_cost * (d^2 + g^2) - predation)
        end
        for j in 1:a.predators
            i = a.prey + j
            h, v = u[2i - 1], u[2i]
            intake = sum(attack(j, k) * x[k] for k in 1:a.prey) / saturation[j]
            dx[i] = x[i] * (a.conversion * intake - a.mortality[j] / (1 + v) - a.crowding * x[i] - a.strategy_cost * (h^2 + v^2))
        end
        return nothing
    end

    function dynamics!(dx, x, p, s, a)
        n = player_number(a)
        q = physical_number(a)
        duration = p[end]
        rates!(dx, x, p, a)
        for i in 1:q
            dx[i] *= duration
        end
        # Cycle averages: integrals over normalized time carry no duration factor.
        for i in 1:n
            effort = p[2i - 1]^2 + p[2i]^2
            features = (-x[i] / a.scale[i], effort, (x[i] / a.scale[i] - 1)^2)
            dx[q + i] = sum(a.weights[i, k] * features[k] for k in 1:3)
            dx[q + n + i] = effort
        end
        return nothing
    end

    objective(x, i, p, a) = x[physical_number(a) + i]

    function constraints(a)
        n = player_number(a); q = physical_number(a)
        bounds = StateConstraint[]
        for i in 1:n
            a.budgets[i] === nothing || push!(
                bounds, StateConstraint(;
                    player = i, index = q + n + i,
                    label = "mean squared effort u$(subscript(2i - 1)), u$(subscript(2i))", upper = a.budgets[i]
                )
            )
        end
        for i in 1:n
            a.floors[i] === nothing || push!(bounds, StateConstraint(; player = i, index = i, label = "floor x$(subscript(i))", lower = a.floors[i], path = true))
        end
        append!(bounds, periodic_constraints(a.x0, n; tolerance = a.periodicity_tolerance))
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
        all(>(0), a.x0) && all(>(0), a.scale) || error("Positive population references required")
        a.prey >= 2 && a.predators >= 2 || error("At least two prey and two predators required")
        length(a.r) == length(a.K) == a.prey && length(a.mortality) == a.predators || error("Species parameter dimensions disagree")
        size(a.attack) == (a.predators, a.prey) && all(>(0), a.K) || error("Invalid trophic matrix or capacities")
        length(a.x0) == length(a.scale) == q && length(a.initial) == control_number(a) || error("State/control dimensions disagree")
        length(a.budgets) == n && size(a.weights) == (n, 3) || error("Budget or weight dimensions disagree")
        indices = collect(1:n)
        length(a.floors) == length(indices) || error("Floor dimensions disagree")
        for name in fieldnames(Parameters)
            name in (:budgets, :floors, :default_size) && continue
            value = getfield(a, name)
            values = value isa Number ? (value,) : value
            all(isfinite, values) && all(>=(0), values) || error("Parameters must be finite and nonnegative: $name")
        end
        all(a.initial .<= 1) || error("Invalid initial strategy")
        isfinite(a.periodicity_tolerance) && a.periodicity_tolerance < minimum(a.x0) || error("Invalid cycle return tolerance")
        0 < a.duration_bounds[1] <= a.duration <= a.duration_bounds[2] < Inf || error("A positive, bounded cycle duration is required")
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
            name = "PredatorPrey", state_labels = vcat(["X$(subscript(i)): biomass prey" for i in 1:a.prey], ["Y$(subscript(j)): biomass predator" for j in 1:a.predators]),
            control_labels = vcat(["u$(subscript(2i - 2 + k)): $meaning" for i in 1:a.prey for (k, meaning) in enumerate(("prey defense", "prey growth"))], ["u$(subscript(2 * (a.prey + j) - 2 + k)): $meaning" for j in 1:a.predators for (k, meaning) in enumerate(("hunting", "maintenance"))]), owners = repeat(collect(1:n); inner = 2),
            dynamics! = (dx, x, p, s) -> dynamics!(dx, x, p, s, a), objective = (x, i, p) -> objective(x, i, p, a),
            constraints = constraints(a), x0 = vcat(a.x0, zeros(2n)), T = 1.0, quadratures = collect((q + 1):(q + 2n)),
            running_objectives = collect((q + 1):(q + n)), lower = zeros(nc), upper = ones(nc), initial = copy(a.initial),
            parameters = [a.duration], parameter_owners = [1], parameter_labels = ["duration"],
            bounds_p = ([a.duration_bounds[1]], [a.duration_bounds[2]]), time_parameter = 1, metadata = a
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
    CorleoneGame.run_logged(PredatorPrey.main, @__DIR__)
end
