# Category: seasonal, population, antagonism
# Leaf and egg depletion precede omitted leaf renewal and reproduction.
# Forward Tupelo/leafminer case specification; the defaults map scenario 11 of
# Low, Ellner & Holden (2013); tupelo_allowance.jl defines the alternative of Section 5.3.
using CorleoneGame
module Tupelo
    using CorleoneGame

    Base.@kwdef struct Parameters
        # Reference instance: scenario 11 of Low, Ellner & Holden (2013), Table 2
        # (lethality 0.5/day, parasitized fraction 0.5, larval life span 25 days),
        # mapped to one time unit = 25 days, so that T = 8 corresponds to 200 days.
        T::Float64 = 8.0
        L0::Float64 = 1.0
        E0::Float64 = 1.0
        H0::Float64 = 0.0
        b::Float64 = 1.5          # 0.75 (pupation + mortality)
        K::Float64 = 0.15         # used only with saturation = true
        muE::Float64 = 0.0        # egg mortality neglected
        muH::Float64 = 1.0        # background larval mortality
        dE::Float64 = 0.25        # defense effect on eggs
        dH::Float64 = 12.5        # defense effect on larvae
        kappa::Float64 = 1.0      # pupation rate
        p_decay::Float64 = 0.2   # used only with photoperiod = false
        q_decay::Float64 = 0.125  # 1/sigma, sigma = 200 days
        t_star::Float64 = 6.0     # photoperiod scale (our assumption; not given by Low et al.)
        z::Float64 = 9.0          # photoperiod steepness
        umax::Float64 = 1.0
        vmax::Float64 = 1.0
        w_A::NTuple{4, Float64} = (0.49, 0.49, 0.02, 0.0)
        w_B::NTuple{3, Float64} = (0.98, 0.02, 0.0)
        scales_P::NTuple{4, Float64} = (1.0, 1.0, 1.0, 1.0)
        scales_H::NTuple{3, Float64} = (1.0, 1.0, 1.0)
        U_A::Union{Nothing, Float64} = nothing # nothing disables
        U_B::Union{Nothing, Float64} = nothing
        C_A::Union{Nothing, Float64} = nothing
        saturation::Bool = false  # true: rho(L) = L/(K+L) in feeding and pupation
        photoperiod::Bool = true  # true: leaf value exp(-(t/t_star)^z); false: exp(-p_decay t)
    end

    function algorithm_options(; kwargs...)
        # Damping 0.2: undamped or weakly damped sweeps oscillate in this high-lethality regime.
        defaults = (; shooting = :single, nlp_solver = :auto, damping = 0.2)
        return Options(; merge(defaults, (; kwargs...))...)
    end

    leaf_value(t, a) = a.photoperiod ? exp(-(t / a.t_star)^a.z) : exp(-a.p_decay * t)

    function dynamics!(dx, x, controls, t, a)
        L, E, H = x[1:3]
        u, v = controls
        g = a.saturation ? L / (a.K + L) : one(L)
        dx[1] = -a.b * g * H
        dx[2] = -(v + a.muE + a.dE * u) * E
        dx[3] = v * E - (a.muH + a.dH * u + a.kappa * g) * H
        wp = a.w_A ./ a.scales_P
        wh = a.w_B ./ a.scales_H
        dx[4] = -(wp[1] * leaf_value(t, a) - wp[2] * u - wp[3] * u^2) * L
        dx[5] = -wh[1] * exp(-a.q_decay * t) * a.kappa * g * H + wh[2] * v^2 * E
        dx[6] = u
        dx[7] = v
        dx[8] = u * L
        return nothing
    end

    function objective(x, player, a)
        return player == 1 ? x[4] - a.w_A[4] / a.scales_P[4] * x[1] :
            x[5] - a.w_B[3] / a.scales_H[3] * x[3]
    end

    function constraints(a)
        bounds = StateConstraint[]
        for (player, index, bound, label) in ((1, 6, a.U_A, "budget u₁"), (2, 7, a.U_B, "budget u₂"), (1, 8, a.C_A, "carbon budget"))
            bound === nothing || push!(bounds, StateConstraint(; player, index, label, upper = bound))
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
        for f in fieldnames(Parameters)
            value = getfield(a, f)
            isnothing(value) && continue
            vals = value isa Tuple ? value : (value,)
            all(x -> isfinite(x) && x >= 0, vals) || error("Invalid parameter $f")
        end
        for (w, scales) in ((a.w_A, a.scales_P), (a.w_B, a.scales_H))
            isapprox(sum(w), 1.0; atol = 1.0e-12) || error("Weights must sum to one")
            all(>(0), scales) || error("Feature scales must be positive")
        end
        return minimum((a.T, a.L0, a.E0, a.K, a.umax, a.vmax)) > 0 || error("Invalid scale or capacity")
    end

    function validate_trajectory(x, a)
        minimum(x[1:3, :]) >= -1.0e-8 || error("Negative biological state")
        maximum(diff(x[1, :])) <= 1.0e-8 || error("Leaf area increased")
        return maximum(diff(vec(x[2, :] + x[3, :]))) <= 1.0e-8 || error("Cohort mass increased")
    end

    function game(a = Parameters())
        validate_parameters(a)
        return Game(;
            name = "Tupelo",
            state_labels = ["L: leaf area", "E: eggs", "H: larvae"],
            control_labels = ["u₁: plant defense", "u₂: hatching hazard"], owners = [1, 2],
            dynamics! = (dx, x, u, t) -> dynamics!(dx, x, u, t, a),
            objective = (x, i) -> objective(x, i, a),
            constraints = constraints(a),
            x0 = [a.L0, a.E0, a.H0, 0, 0, 0, 0, 0], T = a.T, quadratures = collect(4:8),
            lower = [0.0, 0.0], upper = [a.umax, a.vmax], initial = [0.2a.umax, 0.4a.vmax],
            state_lower = [0, 0, 0, -Inf, -Inf, -Inf, -Inf, -Inf],
            running_objectives = [4, 5],
            validate = x -> validate_trajectory(x, a),
            metadata = a
        )
    end

    function main(;
            shooting = :single, parameters = Parameters(),
            options = algorithm_options(; shooting), folder = joinpath(@__DIR__, options.shooting == :single ? "output" : "output_multiple"), synthetic = false
        )
        g = game(parameters)
        synthetic && return create_synthetic_data(g, folder)
        return run_game(g; options, folder)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    CorleoneGame.run_logged(Tupelo.main, @__DIR__)
end
