# Category: periodical, cellular, antagonism
# A resource-driven consumer–prey cycle closes populations and expressed phenotypes.
# Bacterium-ciliate benchmark; uncalibrated values.
using CorleoneGame
module Ciliate
    using CorleoneGame

    # Synthetic periodic extension: fixed phase-zero data were prepared with constant
    # reference controls (tight cycle integration).
    # Custom kinetics, periods, or reference controls require new compatible fixed x0.
    # Approximate closure is not a claim of long-run orbital stability or equilibrium.
    Base.@kwdef struct Parameters
        # External period is fixed data; T spans whole periods, never a decision variable.
        forcing_period::Float64 = 20.0
        T::Float64 = forcing_period
        forcing_amplitude::Float64 = 0.8 # Pronounced seasonal bacterial productivity.
        # Absolute return band; with 1e-4 the best-response iteration creeps (tested 2026-09-26).
        periodicity_tolerance::Float64 = 1.0e-2
        r::Float64 = 1.0
        K::Float64 = 1.0
        c_A::Float64 = 0.15
        a0::Float64 = 2.0
        alpha::Float64 = 2.0
        h::Float64 = 0.5
        e::Float64 = 0.5
        m_B::Float64 = 0.15
        c_B::Float64 = 0.1
        x0::NTuple{2, Float64} = (0.5055436643883756, 0.3564873025218564)
        w_A::NTuple{3, Float64} = (0.9, 0.08, 0.02)
        w_B::NTuple{3, Float64} = (0.4, 0.5, 0.1)
        scales_A::NTuple{3, Float64} = (1.0, 1.0, 1.0)
        scales_B::NTuple{3, Float64} = (1.0, 1.0, 1.0)
        U_A::Union{Nothing, Float64} = 0.1 * T # Per-cycle squared effort budget.
        U_B::Union{Nothing, Float64} = 0.3 * T
        epsilon_A::Union{Nothing, Float64} = nothing
        epsilon_B::Union{Nothing, Float64} = nothing
        initial::NTuple{2, Float64} = (0.2, 0.5)
        phenotype::Bool = true
        tau_A::Float64 = 0.75
        tau_B::Float64 = 0.25
        phenotype0::NTuple{2, Float64} = (0.2, 0.5)
    end

    "Case-specific algorithm defaults; keyword arguments override these settings."
    function algorithm_options(; kwargs...)
        defaults = (; shooting = :single, nlp_solver = :ipopt)
        return Options(; merge(defaults, (; kwargs...))...)
    end

    # Numerical states: log densities, optional phenotypes, then costs and efforts.
    function dynamics!(dx, x, u, t, a)
        bacteria, ciliate = exp(x[1]), exp(x[2])
        defense, feeding = a.phenotype ? (x[3], x[4]) : (u[1], u[2])
        capture = a.a0 * feeding * exp(-a.alpha * defense)
        rate = capture * bacteria / (1 + a.h * capture * bacteria)
        # F/x₁ is evaluated algebraically to avoid dividing tiny densities.
        growth = a.r * (1 + a.forcing_amplitude * sin(2pi * t / a.forcing_period))
        dx[1] = growth * (1 - bacteria / a.K) - a.c_A * defense^2 - capture / (1 + a.h * capture * bacteria) * ciliate
        dx[2] = a.e * rate - a.m_B - a.c_B * feeding^2
        q = a.phenotype ? 4 : 2
        if a.phenotype
            dx[3] = (u[1] - x[3]) / a.tau_A
            dx[4] = (u[2] - x[4]) / a.tau_B
        end
        wa, wb = a.w_A ./ a.scales_A, a.w_B ./ a.scales_B
        # Average maintained density and gross production replace net log expansion.
        dx[q + 1] = (-wa[1] * bacteria / a.x0[1] - wa[2] * growth * bacteria + wa[3] * u[1]^2) / a.T
        dx[q + 2] = (-wb[1] * ciliate / a.x0[2] - wb[2] * a.e * rate * ciliate + wb[3] * u[2]^2) / a.T
        dx[q + 3] = u[1]^2
        dx[q + 4] = u[2]^2
        return nothing
    end

    function objective(x, player, a)
        q = a.phenotype ? 4 : 2
        return x[q + player]
    end

    function constraints(a)
        q = a.phenotype ? 4 : 2
        bounds = StateConstraint[]
        for (player, budget, floor) in ((1, a.U_A, a.epsilon_A), (2, a.U_B, a.epsilon_B))
            budget === nothing || push!(bounds, StateConstraint(; player, index = q + 2 + player, label = "squared budget u$(subscript(player))", upper = budget))
            floor === nothing || push!(bounds, StateConstraint(; player, index = player, label = "floor x$(subscript(player))", lower = log(floor), path = true))
        end
        physical0 = vcat(collect(a.x0), a.phenotype ? collect(a.phenotype0) : Float64[])
        append!(bounds, periodic_constraints(physical0, 2; tolerance = a.periodicity_tolerance, log_indices = [1, 2]))
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
        for f in fieldnames(Parameters)
            value = getfield(a, f)
            value === nothing && continue
            all(x -> isfinite(x)&&x >= 0, value isa Tuple ? value : (value,)) || error("Invalid parameter $f")
        end
        min(a.T, a.K, a.h, a.tau_A, a.tau_B, a.x0...) > 0 || error("Invalid positive parameter")
        all(x -> 0 <= x <= 1, a.phenotype0) && all(x -> 0 <= x <= 1, a.initial) || error("Invalid phenotype/control")
        for (w, s) in ((a.w_A, a.scales_A), (a.w_B, a.scales_B))
            isapprox(sum(w), 1; atol = 1.0e-12) && all(>(0), s) || error("Invalid weights or scales")
        end
        for (floor, x0) in ((a.epsilon_A, a.x0[1]), (a.epsilon_B, a.x0[2]))
            floor === nothing || 0 < floor <= x0 || error("Density floor must be positive and initially feasible")
        end
        return
    end

    function validate_trajectory(x, a)
        minimum(x[1:2, :]) > 0 || error("Nonpositive density")
        return if a.phenotype
            minimum(x[3:4, :]) >= -1.0e-8 && maximum(x[3:4, :]) <= 1 + 1.0e-8 || error("Phenotype bounds violated")
        end
    end

    function game(a = Parameters())
        validate_parameters(a)
        q = a.phenotype ? 4 : 2
        labels = ["x₁: bacteria", "x₂: ciliate"]
        a.phenotype && append!(labels, ["x₃: defense", "x₄: feeding"])
        lb, ub = fill(-Inf, q + 4), fill(Inf, q + 4)
        if a.phenotype
            lb[3:4] .= 0; ub[3:4] .= 1
        end
        return Game(;
            name = "Ciliate",
            state_labels = labels,
            control_labels = ["u₁: defense", "u₂: feeding"], owners = [1, 2],
            dynamics! = (dx, x, u, t) -> dynamics!(dx, x, u, t, a),
            objective = (x, i) -> objective(x, i, a),
            constraints = constraints(a),
            x0 = vcat(log.(collect(a.x0)), a.phenotype ? collect(a.phenotype0) : Float64[], zeros(4)), T = a.T, quadratures = collect((q + 1):(q + 4)),
            lower = zeros(2), upper = ones(2), initial = collect(a.initial),
            state_lower = lb,
            state_upper = ub,
            running_objectives = [q + 1, q + 2],
            decode = x -> vcat(exp.(x[1:2]), x[3:end]),
            validate = x -> validate_trajectory(x, a),
            metadata = a
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
    CorleoneGame.run_logged(Ciliate.main, @__DIR__)
end
