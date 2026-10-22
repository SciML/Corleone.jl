# Category: periodical, molecular, regulation
# Recurring energetic supply organizes maintenance; all biological pools return.
# Mitochondria dynamic-game case specification.
using CorleoneGame
module Mitochondria
    using CorleoneGame

    # Synthetic periodic extension: fixed phase-zero data were prepared with constant
    # reference controls (tight cycle integration).
    # Custom kinetics, periods, or reference controls require new compatible fixed x0.
    # Approximate closure is not a claim of long-run orbital stability or equilibrium.
    Base.@kwdef struct Parameters
        initial::NTuple{2, Float64} = (0.2, 0.3)
        # External period is fixed data; T spans whole periods, never a decision variable.
        forcing_period::Float64 = 40.0
        T::Float64 = forcing_period
        forcing_amplitude::Float64 = 0.9 # Strong activity/rest contrast.
        # Absolute return band. With 1e-4 or 1e-3 all shared return conditions are active for
        # every player and sequential best responses creep along a continuum of generalized
        # equilibria without converging within the round limits (tested 2026-09-24).
        periodicity_tolerance::Float64 = 1.0e-2
        # Low basal synthesis: biogenesis is driven by the energetic pool.
        s::NTuple{2, Float64} = (0.1, 0.1)
        amp::NTuple{2, Float64} = (5.0, 4.0)
        d::NTuple{2, Float64} = (0.1, 0.12)
        kf::Float64 = 0.08
        kr::Float64 = 0.3
        dd::Float64 = 0.05
        v::NTuple{2, Float64} = (0.5, 0.45)
        dv::Float64 = 0.5
        eta::Float64 = 0.1
        x0::NTuple{5, Float64} = (3.1288670356712465, 3.1240155532488996, 2.066994657864377, 2.0163420842114115, 0.8547536323004257)
        w_A::NTuple{3, Float64} = (0.7, 0.2, 0.1)
        w_B::NTuple{3, Float64} = (0.45, 0.4, 0.15)
        # Per-cycle effort allowances; the default rates below multiply the fixed horizon.
        budgets::NTuple{2, Union{Nothing, Float64}} = (0.05 * T, 0.1 * T)
        floor::Union{Nothing, Float64} = nothing
    end

    function algorithm_options(; kwargs...)
        defaults = (; shooting = :single, nlp_solver = :ipopt)
        return Options(; merge(defaults, (; kwargs...))...)
    end

    function dynamics!(dx, x, controls, t, a)
        u = controls
        dx[1] = a.s[1] + a.amp[1] * u[1] * x[5] / (1 + x[5]) - a.d[1] * x[1] - a.kf * x[1]^2 + a.kr * x[3]
        dx[2] = a.s[2] + a.amp[2] * u[2] * x[5] / (1 + x[5]) - a.d[2] * x[2] - a.kf * x[2]^2 + a.kr * x[4]
        dx[3] = a.kf * x[1]^2 - (a.kr + a.dd) * x[3]
        dx[4] = a.kf * x[2]^2 - (a.kr + a.dd) * x[4]
        # Repeated activity/rest supply to the shared energetic pool.
        supply = (a.v[1] + a.v[2]) * (1 + a.forcing_amplitude * sin(2pi * t / a.forcing_period))
        dx[5] = supply - a.dv * x[5] - a.eta * (x[3] + x[4] + u[1] + u[2]) * x[5]
        dx[6] = (-a.w_A[1] * x[1] + a.w_A[2] * u[1]^2 + a.w_A[3] * x[3]^2) / a.T
        dx[7] = (-a.w_B[1] * x[2] + a.w_B[2] * u[2]^2 + a.w_B[3] * x[4]^2) / a.T
        dx[8] = u[1]^2
        dx[9] = u[2]^2
        return nothing
    end

    function objective(x, player, a)
        return x[5 + player]
    end

    function constraints(a)
        bounds = [
            StateConstraint(; player = i, index = 7 + i, label = "squared budget u$(subscript(i))", upper = a.budgets[i])
                for i in 1:2 if a.budgets[i] !== nothing
        ]
        a.floor === nothing || append!(bounds, [StateConstraint(; player = i, index = 1, label = "floor x$(subscript(1))", lower = a.floor, path = true) for i in 1:2])
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
            name = "Mitochondria", state_labels = ["M₁: healthy mass A", "M₂: healthy mass B", "D₁: damaged mass A", "D₂: damaged mass B", "V: energetic capacity"], control_labels = ["u₁: mitochondrial repair", "u₂: mitochondrial repair"], owners = [1, 2],
            dynamics! = (dx, x, u, t) -> dynamics!(dx, x, u, t, a), objective = (x, i) -> objective(x, i, a),
            constraints = constraints(a),
            x0 = vcat(collect(a.x0), zeros(4)), T = a.T, quadratures = collect(6:9),
            lower = zeros(2), upper = ones(2), initial = collect(a.initial),
            running_objectives = collect(6:7), metadata = a
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
    CorleoneGame.run_logged(Mitochondria.main, @__DIR__)
end
