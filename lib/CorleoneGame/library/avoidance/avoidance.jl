# Category: episodical, organismal, regulation
# One warning-and-recovery trial starts from a prescribed state.
# Synthetic adaptive-avoidance game.
using CorleoneGame
module Avoidance
    using CorleoneGame

    Base.@kwdef struct Parameters
        T::Float64 = 12.0
        aq::Float64 = 0.8
        bq::Float64 = 0.25
        az::Float64 = 0.9
        dz::Float64 = 0.35
        gamma::Float64 = 0.45
        ar::Float64 = 0.6
        r0::Float64 = 1.0
        cr::Float64 = 0.12
        hr::Float64 = 0.08
        Kr::Float64 = 0.35
        x0::NTuple{4, Float64} = (0.0, 0.0, 0.0, 1.0)
        initial::NTuple{2, Float64} = (0.2, 0.3)
        weights_A::NTuple{4, Float64} = (0.65, 0.15, 0.1, 0.1)
        weights_B::NTuple{4, Float64} = (0.25, 0.45, 0.1, 0.2)
    end

    function algorithm_options(; kwargs...)
        return Options(; merge((; damping = 0.65, warm_start = true), (; kwargs...))...)
    end

    warning_cue(t, a) = t <= a.T / 2 ? 1.0 : 0.0

    function validate_parameters(a)
        all(isfinite, (a.T, a.aq, a.bq, a.az, a.dz, a.gamma, a.ar, a.r0, a.cr, a.hr, a.Kr, a.x0..., a.initial..., a.weights_A..., a.weights_B...)) ||
            error("Avoidance parameters must be finite")
        all(>=(0), (a.aq, a.bq, a.az, a.dz, a.gamma, a.ar, a.r0, a.cr, a.hr, a.Kr)) || error("Invalid rates")
        all(>=(0), a.x0) && all(0 .<= a.initial .<= 1) || error("Invalid initial states or controls")
        for w in (a.weights_A, a.weights_B)
            all(>=(0), w) && isapprox(sum(w), 1; atol = 1.0e-12) || error("Weights must be nonnegative and sum to one")
        end
        return
    end

    function dynamics!(dx, x, u, t, a)
        q, zA, zB, r = x[1:4]
        total = u[1] + u[2]
        dx[1] = a.aq * (warning_cue(t, a) - q) - a.bq * total * q
        dx[2] = a.az * q * (1 - zA) - a.dz * zA + a.gamma * u[1] * r
        dx[3] = a.az * q * (1 - zB) - a.dz * zB + a.gamma * u[2] * r
        dx[4] = a.ar * (a.r0 - r) - a.cr * total - a.hr * (zA + zB)
        w = a.weights_A
        dx[5] = (-w[1] * zA + w[2] * u[1]^2 + w[3] * a.Kr / (a.Kr + max(r, 0.0)) - w[4] * zB) / a.T
        w = a.weights_B
        dx[6] = (-w[1] * zB + w[2] * u[2]^2 + w[3] * a.Kr / (a.Kr + max(r, 0.0)) - w[4] * zA) / a.T
        return nothing
    end

    function game(a = Parameters())
        validate_parameters(a)
        return Game(;
            name = "Avoidance",
            state_labels = ["q: cue trace", "z₁: readiness A", "z₂: readiness B", "r: motivation", "φ₁: cost", "φ₂: cost"],
            control_labels = ["u₁: avoidance intensity", "u₂: avoidance intensity"], owners = [1, 2],
            dynamics! = (dx, x, u, t) -> dynamics!(dx, x, u, t, a),
            objective = (x, i) -> i == 1 ? x[5] : x[6],
            x0 = vcat(collect(a.x0), zeros(2)), T = a.T, quadratures = collect(5:6),
            lower = zeros(2), upper = ones(2), initial = collect(a.initial),
            state_lower = vcat(zeros(4), fill(-Inf, 2)),
            running_objectives = [5, 6], gain_scales = [1.0, 1.0],
            validate = x -> (minimum(x[1:4, :]) >= -1.0e-8 || error("Avoidance state became negative")), metadata = a
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
    CorleoneGame.run_logged(Avoidance.main, @__DIR__)
end
