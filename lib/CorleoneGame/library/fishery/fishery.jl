# Category: periodical, population, competition
# A recurring harvest rotation closes fish biomass, not catch accounts; its duration belongs to A.
# Two-harvester game: Suri (2008), p. 168; synthetic benchmark parameters.
# Run from lib/CorleoneGame: julia --project=. library/fishery/fishery.jl
using CorleoneGame
module Fishery
    using CorleoneGame

    # Autonomous rotation: there is no external clock, so the cycle duration can be a
    # decision of fleet A. Dynamics are written on normalized time s in [0,1] and
    # multiplied by the duration; physical time is duration*s.
    # Default x0 is the stock at which the reference efforts are stationary, so the
    # reference policy closes for every duration. Per-cycle catch quotas make short
    # rotations profitable, which requires a nonstationary depletion-recovery cycle.
    # Approximate closure is not a claim of long-run orbital stability or equilibrium.
    Base.@kwdef struct Parameters
        periodicity_tolerance::Float64 = 1.0e-4
        r::Float64 = 0.8
        K::Float64 = 1.0
        q::NTuple{2, Float64} = (0.7, 0.55)
        price::NTuple{2, Float64} = (1.2, 1.0)
        linear_cost::NTuple{2, Float64} = (0.06, 0.08)
        quadratic_cost::NTuple{2, Float64} = (0.5, 0.65)
        w_A::NTuple{2, Float64} = (0.8, 0.2)
        w_B::NTuple{2, Float64} = (0.6, 0.4) # A competing fleet keeps the rotation contested.
        umax::NTuple{2, Float64} = (1.0, 1.0)
        initial::NTuple{2, Float64} = (0.2, 0.07)
        x0::Float64 = K * (1 - sum(q .* initial) / r) # Stationary stock under the reference efforts.
        # A owns the rotation length. Objectives are average cost rates per unit time.
        duration::Float64 = 6.0
        duration_bounds::NTuple{2, Float64} = (1.0, 12.0)
        # Per-rotation catch totals are accounts, not biological states to be closed.
        quotas::NTuple{2, Union{Nothing, Float64}} = (0.8, 0.4)
        floor::Union{Nothing, Float64} = 0.4
    end

    function algorithm_options(; kwargs...)
        # Undamped responses: averaging controls and duration breaks cycle closure.
        defaults = (; shooting = :single, nlp_solver = :ipopt, ode_tolerance = 1.0e-10, damping = 1.0)
        return Options(; merge(defaults, (; kwargs...))...)
    end

    function dynamics!(dx, x, p, s, a)
        duration = p[3]
        biomass = x[1]
        dx[1] = duration * (a.r * biomass * (1 - biomass / a.K) - biomass * sum(a.q[i] * p[i] for i in 1:2))
        for i in 1:2
            w = i == 1 ? a.w_A : a.w_B
            catch_rate = a.q[i] * p[i] * biomass
            cost = a.linear_cost[i] * p[i] + a.quadratic_cost[i] * p[i]^2 / 2
            # Average over the rotation: integral over normalized time, no duration factor.
            dx[1 + i] = -w[1] * a.price[i] * catch_rate + w[2] * cost
            dx[3 + i] = duration * catch_rate
        end
        return nothing
    end

    objective(x, i, p, a) = x[1 + i]

    function constraints(a)
        bounds = StateConstraint[]
        for i in 1:2
            a.quotas[i] === nothing || push!(
                bounds, StateConstraint(;
                    player = i, index = 3 + i,
                    label = "catch quota", upper = a.quotas[i]
                )
            )
        end
        for i in 1:2
            a.floor === nothing || push!(bounds, StateConstraint(; player = i, index = 1, label = "floor x$(subscript(1))", lower = a.floor, path = true))
        end
        append!(bounds, periodic_constraints([a.x0], 2; tolerance = a.periodicity_tolerance))
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
        all(isfinite, (a.r, a.K, a.x0)) && min(a.r, a.K, a.x0) > 0 || error("Positive finite growth, capacity, and initial stock required")
        for v in (a.q, a.price, a.linear_cost, a.quadratic_cost, a.umax, a.initial)
            all(isfinite, v) && all(>=(0), v) || error("Rates, costs, and efforts must be finite and nonnegative")
        end
        all(>(0), a.umax) && all(a.initial .<= a.umax) || error("Initial effort must lie within positive effort bounds")
        for w in (a.w_A, a.w_B)
            all(isfinite, w) && all(>=(0), w) && isapprox(sum(w), 1; atol = 1.0e-12) || error("Weights must be nonnegative and sum to one")
        end
        for quota in a.quotas
            quota === nothing || (isfinite(quota) && quota >= 0) || error("Invalid catch quota")
        end
        a.floor === nothing || (isfinite(a.floor) && 0 <= a.floor <= a.x0) || error("Stock floor must be initially feasible")
        isfinite(a.periodicity_tolerance) && 0 <= a.periodicity_tolerance < a.x0 || error("Invalid cycle return tolerance")
        return 0 < a.duration_bounds[1] <= a.duration <= a.duration_bounds[2] < Inf || error("A positive, bounded rotation duration is required")
    end

    function game(a = Parameters())
        validate_parameters(a)
        return Game(;
            name = "Fishery", state_labels = ["x: fish biomass"],
            control_labels = ["u₁: fleet effort", "u₂: fleet effort"], owners = [1, 2],
            dynamics! = (dx, x, p, s) -> dynamics!(dx, x, p, s, a), objective = (x, i, p) -> objective(x, i, p, a),
            constraints = constraints(a), x0 = [a.x0, 0, 0, 0, 0], T = 1.0, quadratures = collect(2:5),
            running_objectives = [2, 3], lower = zeros(2), upper = collect(a.umax), initial = collect(a.initial),
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
    CorleoneGame.run_logged(Fishery.main, @__DIR__)
end
