# Category: episodical, organismal, regulation
# One reach changes position; a return movement is not modeled.
# Deterministic motor game; SI units.
using CorleoneGame
module Motor
    using CorleoneGame

    Base.@kwdef struct Parameters
        T::Float64 = 0.5
        m::NTuple{2, Float64} = (2.0, 2.0)
        k::Float64 = 100.0
        b::Float64 = 10.0
        tau::Float64 = 0.1
        s_T::Float64 = 0.005
        s_v::Float64 = 0.005
        s_V::Float64 = 0.0025
        s_d::Float64 = 0.025
        s_u::Float64 = 15.0
        target::NTuple{2, Float64} = (0.0, 0.1)
        via_A::NTuple{2, Float64} = (-0.02, 0.05)
        via_B::NTuple{2, Float64} = (0.02, 0.05)
        via_times::NTuple{2, Float64} = (0.2, 0.4)
        w_A::NTuple{5, Float64} = (0.2, 0.2, 0.2, 0.2, 0.2)
        w_B::NTuple{5, Float64} = (0.2, 0.2, 0.2, 0.2, 0.2)
        umax::Float64 = 30.0
        x0::Vector{Float64} = zeros(16)
        initial::NTuple{4, Float64} = (0.0, 0.0, 0.0, 0.0)
    end

    function algorithm_options(; kwargs...)
        return Options(; merge((; max_iters = 300, warm_start = true), (; kwargs...))...)
    end

    function validate_parameters(a)
        for f in fieldnames(Parameters)
            v = getfield(a, f)
            all(isfinite, v isa Number ? (v,) : v) || error("Nonfinite $f")
        end
        all(>(0), (a.T, a.m..., a.tau, a.s_T, a.s_v, a.s_V, a.s_d, a.s_u, a.umax)) || error("Invalid positive scale")
        min(a.k, a.b) >= 0 || error("Negative stiffness or damping")
        length(a.x0) == 16 || error("Motor requires 16 initial physical states")
        all(t -> 0 < t < a.T, a.via_times) || error("Via times must be interior")
        all(u -> abs(u) <= a.umax, a.initial) || error("Invalid initial commands")
        for w in (a.w_A, a.w_B)
            all(>=(0), w) && isapprox(sum(w), 1; atol = 1.0e-12) || error("Invalid weights")
        end
        return
    end

    # For each player: px, py, vx, vy, fx, fy, ax, ay; then two running costs.
    function dynamics!(dx, x, u, t, a)
        for i in 1:2
            o = 8(i - 1); j = 8(2 - i); c = 2(i - 1)
            for d in 1:2
                dx[o + d] = x[o + 2 + d]
                dx[o + 2 + d] = (x[o + 4 + d] + a.k * (x[j + d] - x[o + d]) - a.b * x[o + 2 + d]) / a.m[i]
                dx[o + 4 + d] = x[o + 6 + d]
                dx[o + 6 + d] = (u[c + d] - x[o + 4 + d] - 2a.tau * x[o + 6 + d]) / a.tau^2
            end
            w = i == 1 ? a.w_A : a.w_B
            dx[16 + i] = (
                w[4] * sum((x[o + d] - x[j + d])^2 for d in 1:2) / a.s_d^2 +
                    w[5] * sum(u[c + d]^2 for d in 1:2) / a.s_u^2
            ) / a.T
        end
        return nothing
    end

    function objective(x, i, a)
        o = 8(i - 1); w = i == 1 ? a.w_A : a.w_B
        return x[16 + i] + w[1] * sum((x[o + d] - a.target[d])^2 for d in 1:2) / a.s_T^2 +
            w[2] * sum(x[o + 2 + d]^2 for d in 1:2) / a.s_v^2
    end

    function game(a = Parameters())
        validate_parameters(a)
        points = PointCost[]
        for i in 1:2
            o = 8(i - 1); w = i == 1 ? a.w_A : a.w_B; via = i == 1 ? a.via_A : a.via_B
            push!(
                points, PointCost(;
                    player = i, time = a.via_times[i],
                    cost = x -> w[3] * sum((x[o + d] - via[d])^2 for d in 1:2) / a.s_V^2
                )
            )
        end
        labels = ["$(s)$(subscript(i))$(d)" for i in 1:2 for s in ("p", "v", "f", "a") for d in ("ₓ", "ᵧ")]
        return Game(;
            name = "Motor",
            state_labels = labels,
            control_labels = ["u₁: horizontal motor command", "u₂: vertical motor command", "u₃: horizontal motor command", "u₄: vertical motor command"], owners = [1, 1, 2, 2],
            dynamics! = (dx, x, u, t) -> dynamics!(dx, x, u, t, a),
            objective = (x, i) -> objective(x, i, a), point_costs = points,
            x0 = vcat(a.x0, zeros(2)), T = a.T, quadratures = [17, 18],
            lower = fill(-a.umax, 4), upper = fill(a.umax, 4), initial = collect(a.initial),
            running_objectives = [17, 18],
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
    CorleoneGame.run_logged(Motor.main, @__DIR__)
end
