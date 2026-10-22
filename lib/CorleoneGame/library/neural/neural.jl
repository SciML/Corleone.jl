# Category: periodical, organismal, regulation
# A driven excitatory–inhibitory population cycle; all activities return.
# Synthetic Wilson--Cowan regulatory game. Time is ms.
using CorleoneGame
module Neural
    using CorleoneGame

    # Synthetic periodic extension: fixed phase-zero data were prepared with constant
    # reference controls (tight cycle integration).
    # Custom kinetics, periods, or reference controls require new compatible fixed x0.
    # Approximate closure is not a claim of long-run orbital stability or equilibrium.
    Base.@kwdef struct Parameters
        # External period is fixed data; T spans whole periods, never a decision variable.
        forcing_period::Float64 = 100.0
        T::Float64 = forcing_period
        forcing_amplitude::Float64 = 0.2
        periodicity_tolerance::Float64 = 1.0e-4
        c_EE::Float64 = 16.0
        c_EI::Float64 = 12.0
        c_IE::Float64 = 15.0
        c_II::Float64 = 3.0
        a_E::Float64 = 1.3
        a_I::Float64 = 2.0
        b_E::Float64 = 4.0
        b_I::Float64 = 3.7
        tau_E::Float64 = 8.0
        tau_I::Float64 = 8.0
        P::Float64 = 1.0
        Q::Float64 = 0.0
        E_A::Float64 = 0.35
        E_B::Float64 = 0.15
        I_B::Float64 = 0.2
        umax::Float64 = 2.0
        x0::NTuple{2, Float64} = (0.17262880251023957, 0.10727034562050068)
        initial::NTuple{2, Float64} = (0.5, 0.5)
        w_A::NTuple{3, Float64} = (0.8, 0.1, 0.1)
        w_B::NTuple{3, Float64} = (0.4, 0.5, 0.1)
    end

    function algorithm_options(; kwargs...)
        # Empirical stabilization; continuation can still cycle in this nonconvex game.
        return Options(; merge((; max_iters = 500, damping = 0.15, warm_start = false), (; kwargs...))...)
    end

    # Stable for both signs, including automatic-differentiation inputs.
    sigmoid(z) = z >= 0 ? inv(1 + exp(-z)) : exp(z) / (1 + exp(z))

    function validate_parameters(a)
        validate_periodic_clock(a.T, a.forcing_period, a.forcing_amplitude)
        for f in fieldnames(Parameters)
            v = getfield(a, f)
            all(isfinite, v isa Tuple ? v : (v,)) || error("Nonfinite $f")
        end
        all(>(0), (a.T, a.tau_E, a.tau_I, a.a_E, a.a_I, a.umax)) || error("Invalid positive scale")
        all(>=(0), (a.c_EE, a.c_EI, a.c_IE, a.c_II)) || error("Negative coupling magnitude")
        all(x -> 0 <= x <= 1, (a.x0..., a.E_A, a.E_B, a.I_B)) && a.E_A > a.E_B || error("Invalid activities or targets")
        all(x -> 0 <= x <= a.umax, a.initial) || error("Invalid initial drives")
        for w in (a.w_A, a.w_B)
            all(>=(0), w) && isapprox(sum(w), 1; atol = 1.0e-12) || error("Invalid weights")
        end
        return
    end

    function dynamics!(dx, x, u, t, a)
        E, I = x[1], x[2]
        drive = a.P + a.forcing_amplitude * sin(2pi * t / a.forcing_period)
        dx[1] = (-E + (1 - E) * sigmoid(a.a_E * (a.c_EE * E - a.c_EI * I + drive + u[1] - a.b_E))) / a.tau_E
        dx[2] = (-I + (1 - I) * sigmoid(a.a_I * (a.c_IE * E - a.c_II * I + a.Q + u[2] - a.b_I))) / a.tau_I
        dx[3] = ((a.w_A[1] + a.w_A[3]) * (E - a.E_A)^2 + a.w_A[2] * (u[1] / a.umax)^2) / a.T
        dx[4] = (a.w_B[1] * (E - a.E_B)^2 + a.w_B[2] * (u[2] / a.umax)^2 + a.w_B[3] * (I - a.I_B)^2) / a.T
        return nothing
    end

    function game(a = Parameters())
        validate_parameters(a)
        return Game(;
            name = "Neural",
            state_labels = ["E: excitatory activity", "I: inhibitory activity"],
            control_labels = ["u₁: excitatory drive", "u₂: inhibitory drive"], owners = [1, 2],
            dynamics! = (dx, x, u, t) -> dynamics!(dx, x, u, t, a),
            # The former terminal activity penalty is now an average tracking penalty.
            objective = (x, i) -> x[2 + i],
            constraints = periodic_constraints(collect(a.x0), 2; tolerance = a.periodicity_tolerance),
            x0 = vcat(collect(a.x0), zeros(2)), T = a.T, quadratures = [3, 4],
            lower = zeros(2), upper = fill(a.umax, 2), initial = collect(a.initial),
            state_lower = [0, 0, -Inf, -Inf],
            state_upper = [1, 1, Inf, Inf],
            running_objectives = [3, 4],
            validate = x -> (minimum(x[1:2, :]) >= -1.0e-8 && maximum(x[1:2, :]) <= 1 + 1.0e-8 || error("Activity bounds violated")),
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
    CorleoneGame.run_logged(Neural.main, @__DIR__)
end
