# Category: periodical, molecular, regulation
# Recurring substrate supply drives a pathway cycle; all biological pools return.
# MetabolicPathway dynamic-game case specification.
using CorleoneGame
module MetabolicPathway
    using CorleoneGame

    # Synthetic periodic extension: fixed phase-zero data were prepared with constant
    # reference controls (tight cycle integration).
    # Custom kinetics, periods, or reference controls require new compatible fixed x0.
    # Approximate closure is not a claim of long-run orbital stability or equilibrium.
    Base.@kwdef struct Parameters
        initial::NTuple{2, Float64} = (0.2, 0.3)
        # External period is fixed data; T spans whole periods, never a decision variable.
        forcing_period::Float64 = 12.0
        T::Float64 = forcing_period
        forcing_amplitude::Float64 = 0.8 # Strong feed/starve contrast.
        periodicity_tolerance::Float64 = 1.0e-2
        V::NTuple{2, Float64} = (1.2, 1.0)
        K::NTuple{2, Float64} = (0.4, 0.6)
        # Byproduct inhibition constants of the two branches.
        KX::NTuple{2, Float64} = (1.0, 1.0)
        d::NTuple{2, Float64} = (0.1, 0.08)
        D::Float64 = 0.4
        Sin::Float64 = 3.0
        dp::Float64 = 0.2
        al::NTuple{2, Float64} = (0.8, 0.7)
        be::NTuple{2, Float64} = (0.3, 0.25)
        x0::NTuple{6, Float64} = (1.7737523591883528, 2.2612034026403354, 1.0321725845734169, 1.7212385836246837, 0.5333333333333322, 0.8399999999999984)
        w_A::NTuple{2, Float64} = (0.9, 0.1)
        w_B::NTuple{2, Float64} = (0.7, 0.3)
        # Per-cycle effort allowances; the default rates below multiply the fixed horizon.
        budgets::NTuple{2, Union{Nothing, Float64}} = (0.3 * T, 0.25 * T)
        floor::Union{Nothing, Float64} = nothing
    end

    function algorithm_options(; kwargs...)
        defaults = (; shooting = :single, nlp_solver = :ipopt)
        return Options(; merge(defaults, (; kwargs...))...)
    end

    # Productive flux of branch j; its own byproduct X_j inhibits the reaction.
    flux(x, j, a) = a.V[j] * x[4 + j] * x[3] / (a.K[j] + x[3]) / (1 + x[j] / a.KX[j])

    function dynamics!(dx, x, controls, t, a)
        u = controls
        v1, v2 = flux(x, 1, a), flux(x, 2, a)
        dx[1] = v1 - a.d[1] * x[1]
        dx[2] = v2 - a.d[2] * x[2]
        dx[3] = a.D * (a.Sin * (1 + a.forcing_amplitude * sin(2pi * t / a.forcing_period)) - x[3]) - v1 - v2
        dx[4] = v1 + v2 - a.dp * x[4]
        dx[5] = a.al[1] * u[1] - a.be[1] * x[5]
        dx[6] = a.al[2] * u[2] - a.be[2] * x[6]
        dx[7] = (-a.w_A[1] * x[4] + a.w_A[2] * u[1]^2) / a.T
        dx[8] = (-a.w_B[1] * x[4] + a.w_B[2] * u[2]^2) / a.T
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
        a.floor === nothing || append!(bounds, [StateConstraint(; player = i, index = 3, label = "floor x$(subscript(3))", lower = a.floor, path = true) for i in 1:2])
        append!(bounds, periodic_constraints(collect(a.x0), 2; tolerance = a.periodicity_tolerance))
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
        validate_periodic_clock(a.T, a.forcing_period, a.forcing_amplitude)
        for b in a.budgets
            b === nothing || (isfinite(b) && b >= 0) || error("Invalid effort budget")
        end
        a.T > 0 || error("T must be positive")
        all(isfinite, a.x0) || error("Initial states must be finite")
        a.floor === nothing || (isfinite(a.floor) && 0 <= a.floor <= a.x0[3]) ||
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
            name = "MetabolicPathway", state_labels = ["X₁: byproduct A", "X₂: byproduct B", "S: substrate", "Y: product", "E₁: enzyme A", "E₂: enzyme B"], control_labels = ["u₁: enzyme allocation", "u₂: enzyme allocation"], owners = [1, 2],
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
    CorleoneGame.run_logged(MetabolicPathway.main, @__DIR__)
end
