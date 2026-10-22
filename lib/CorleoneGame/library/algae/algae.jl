# Category: periodical, cellular, competition, variablePlayers
# A maintained culture under recurring carbon supply; biomasses and resources return.
# Dimensionless synthetic resource-allocation game.
using CorleoneGame
module Algae
    using CorleoneGame

    # Abstract uptake types, deliberately not mapped to named species.
    const UPTAKE_TYPES = (:HCHH, :HCLH, :LCHH, :LCLH)
    const KC = (1.0, 1.0, 40.0, 40.0)
    const KH = (30.0, 1200.0, 30.0, 1200.0)
    const VC = (0.2, 0.2, 0.4, 0.4)
    const VH = (0.2, 0.4, 0.2, 0.4)

    # Synthetic periodic extension: fixed phase-zero data were prepared with constant
    # reference controls (tight cycle integration).
    # Custom kinetics, periods, or reference controls require new compatible fixed x0.
    # Approximate closure is not a claim of long-run orbital stability or equilibrium.
    Base.@kwdef struct Parameters
        types::Vector{Symbol} = collect(UPTAKE_TYPES)
        # External period is fixed data; T spans whole periods, never a decision variable.
        forcing_period::Float64 = 60.0
        T::Float64 = forcing_period
        forcing_amplitude::Float64 = 0.9 # Strong feed pulses drive bloom and decline.
        periodicity_tolerance::Float64 = 1.0e-4
        B0::Float64 = 1.0
        K_C::Vector{Float64} = [KC[findfirst(==(s), UPTAKE_TYPES)] for s in types]
        K_H::Vector{Float64} = [KH[findfirst(==(s), UPTAKE_TYPES)] for s in types]
        V_C::Vector{Float64} = [VC[findfirst(==(s), UPTAKE_TYPES)] for s in types]
        V_H::Vector{Float64} = [VH[findfirst(==(s), UPTAKE_TYPES)] for s in types]
        Y::Vector{Float64} = ones(length(types))
        # Losses calibrated to balance gross growth over one reference feed cycle.
        d::Vector{Float64} = [(0.11390889313478779, 0.0632729540737821, 0.0673378029866455, 0.016701863925639288)[findfirst(==(s), UPTAKE_TYPES)] for s in types]
        lambda::Float64 = 0.2
        k_plus::Float64 = 0.3
        k_minus::Float64 = 0.01
        C_in::Float64 = 2.0
        H_in::Float64 = 60.0
        # Fixed phase-zero stocks of the maintained reference culture (see the preparation script).
        x0::Vector{Float64} = vcat(fill(1.0, length(types)), 1.0465096857477418, 37.025387624328054)
        # Moderate specialization cost: allocations track the carbon supply without saturating.
        weights::Matrix{Float64} = repeat([0.45 0.1 0.45], length(types), 1)
        initial::Vector{Float64} = fill(0.5, length(types))
    end

    function algorithm_options(; kwargs...)
        return Options(; kwargs...)
    end

    function validate_parameters(a)
        validate_periodic_clock(a.T, a.forcing_period, a.forcing_amplitude)
        n = length(a.types)
        n >= 2 && all(s -> s in UPTAKE_TYPES, a.types) || error("Choose at least two abstract uptake types")
        for key in (:K_C, :K_H, :V_C, :V_H, :Y, :d, :initial)
            v = getfield(a, key)
            length(v) == n && all(isfinite, v) && all(>=(0), v) || error("Invalid $key")
        end
        all(>(0), a.K_C) && all(>(0), a.K_H) && all(>(0), a.Y) || error("Affinities and yields must be positive")
        all(x -> isfinite(x)&&x > 0, (a.T, a.B0)) || error("Invalid horizon or biomass scale")
        all(x -> isfinite(x)&&x >= 0, (a.lambda, a.k_plus, a.k_minus, a.C_in, a.H_in)) || error("Invalid resource parameters")
        length(a.x0) == n + 2 && all(x -> isfinite(x)&&x >= 0, a.x0) || error("Invalid initial states")
        size(a.weights) == (n, 3) && all(x -> isfinite(x)&&x >= 0, a.weights) &&
            all(abs.(sum(a.weights; dims = 2) .- 1) .<= 1.0e-12) || error("Invalid weights")
        return all(<=(1), a.initial) || error("Invalid initial allocations")
    end

    function dynamics!(dx, x, u, t, a)
        n = length(a.types)
        # Extend uptake continuously outside the invariant resource domain for solver trials.
        C, H = max(x[n + 1], 0), max(x[n + 2], 0)
        feed = 1 + a.forcing_amplitude * sin(2pi * t / a.forcing_period)
        dx[n + 1] = a.lambda * (a.C_in * feed - C) - a.k_plus * C + a.k_minus * H
        dx[n + 2] = a.lambda * (a.H_in * feed - H) + a.k_plus * C - a.k_minus * H
        for i in 1:n
            qC = u[i] * a.V_C[i] * C / (a.K_C[i] + C)
            qH = (1 - u[i]) * a.V_H[i] * H / (a.K_H[i] + H)
            dx[i] = (a.Y[i] * (qC + qH) - a.d[i]) * x[i]
            dx[n + 1] -= qC * x[i]
            dx[n + 2] -= qH * x[i]
            dx[n + 2 + i] = (-(a.weights[i, 1] + a.weights[i, 3]) * a.Y[i] * (qC + qH) * x[i] / a.B0 + a.weights[i, 2] * (u[i]^2 + (1 - u[i])^2)) / a.T
        end
        return nothing
    end

    function game(a = Parameters())
        validate_parameters(a); n = length(a.types); q = n + 2
        return Game(;
            name = "Algae N=$n",
            state_labels = vcat(["SCALE 50 X$(subscript(i)): $(a.types[i])" for i in 1:n], ["C: CO₂", "H: bicarbonate"]),
            control_labels = ["u$(subscript(i)): CO₂ allocation" for i in 1:n], owners = collect(1:n),
            dynamics! = (dx, x, u, t) -> dynamics!(dx, x, u, t, a),
            # Average gross biomass production competes with allocation cost;
            # terminal biomass is fixed by closure and is not a fitness feature.
            objective = (x, i) -> x[q + i],
            constraints = periodic_constraints(a.x0, n; tolerance = a.periodicity_tolerance),
            x0 = vcat(a.x0, zeros(n)), T = a.T, quadratures = collect((q + 1):(q + n)),
            lower = zeros(n), upper = ones(n), initial = copy(a.initial),
            state_lower = vcat(zeros(q), fill(-Inf, n)),
            running_objectives = collect((q + 1):(q + n)),
            validate = x -> (minimum(x[1:q, :]) >= -1.0e-8 || error("Negative algae/resource state")),
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
    CorleoneGame.run_logged(Algae.main, @__DIR__)
end
