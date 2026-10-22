# Category: episodical, cellular, antagonism
# One infection experiment changes densities and expressed traits; transfer is external.
# Bacterium-phage benchmark with an infected-cell (latent-period) compartment; uncalibrated values.
using CorleoneGame
module Phage
    using CorleoneGame

    Base.@kwdef struct Parameters
        T::Float64 = 8.0
        r::Float64 = 1.0
        K::Float64 = 1.0
        c_A::Float64 = 0.1
        a0::Float64 = 0.3
        alpha_A::Float64 = 2.0
        alpha_B::Float64 = 1.5
        beta0::Float64 = 4.0
        c_B::Float64 = 0.3
        m_B::Float64 = 0.4
        # Mean latent period between adsorption and lysis.
        latent::Float64 = 0.5
        tau_A::Float64 = 0.75
        tau_B::Float64 = 0.5
        # Susceptible bacteria, infected bacteria, free phage, resistance, infectivity.
        x0::NTuple{5, Float64} = (0.6, 0.02, 0.15, 0.2, 0.2)
        w_A::NTuple{3, Float64} = (0.6, 0.3, 0.1)
        w_B::NTuple{3, Float64} = (0.9, 0.0, 0.1)
        scales_A::NTuple{3, Float64} = (1.0, 1.0, 1.0)
        scales_B::NTuple{3, Float64} = (1.0, 1.0, 1.0)
        # nothing disables a constraint. Floors are disabled in the reference instance.
        U_A::Union{Nothing, Float64} = 1.8
        U_B::Union{Nothing, Float64} = 2.0
        epsilon_A::Union{Nothing, Float64} = nothing
        epsilon_B::Union{Nothing, Float64} = nothing
        umax::NTuple{2, Float64} = (1.0, 1.0)
        initial::NTuple{2, Float64} = (0.2, 0.2)
    end

    function algorithm_options(; kwargs...)
        defaults = (; shooting = :single, nlp_solver = :ipopt)
        return Options(; merge(defaults, (; kwargs...))...)
    end

    adsorption(resistance, infectivity, a) = a.a0 * exp(a.alpha_B * infectivity - a.alpha_A * resistance)
    burst_size(infectivity, a) = 1 + (a.beta0 - 1) * exp(-a.c_B * infectivity^2)

    # Numerical states: three log densities, two phenotypes, two running costs, two efforts.
    function dynamics!(dx, x, u, t, a)
        susceptible, infected, phage = exp(x[1]), exp(x[2]), exp(x[3])
        resistance, infectivity = x[4], x[5]
        ads = adsorption(resistance, infectivity, a)
        burst = burst_size(infectivity, a)
        lysis = infected / a.latent
        dx[1] = a.r * (1 - (susceptible + infected) / a.K) - a.c_A * resistance^2 - ads * phage
        dx[2] = ads * susceptible * phage / infected - 1 / a.latent
        # Free phage are released at lysis and lost by adsorption to any cell and by decay.
        dx[3] = burst * lysis / phage - ads * (susceptible + infected) - a.m_B
        dx[4] = (u[1] - resistance) / a.tau_A
        dx[5] = (u[2] - infectivity) / a.tau_B
        wa, wb = a.w_A ./ a.scales_A, a.w_B ./ a.scales_B
        dx[6] = -wa[2] * a.r * susceptible + wa[3] * u[1]^2
        dx[7] = -wb[2] * burst * lysis + wb[3] * u[2]^2
        dx[8] = u[1]^2
        dx[9] = u[2]^2
        return nothing
    end

    function objective(x, player, a)
        return player == 1 ? x[6] - a.w_A[1] / a.scales_A[1] * (x[1] - log(a.x0[1])) :
            x[7] - a.w_B[1] / a.scales_B[1] * (x[3] - log(a.x0[3]))
    end

    function constraints(a)
        bounds = StateConstraint[]
        for (player, budget, floor, index) in ((1, a.U_A, a.epsilon_A, 1), (2, a.U_B, a.epsilon_B, 3))
            budget === nothing || push!(bounds, StateConstraint(; player, index = 7 + player, label = "squared budget u$(subscript(player))", upper = budget))
            floor === nothing || push!(bounds, StateConstraint(; player, index, label = "floor x$(subscript(index))", lower = log(floor), path = true))
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
            value === nothing && continue
            all(x -> isfinite(x)&&x >= 0, value isa Tuple ? value : (value,)) || error("Invalid parameter $f")
        end
        min(a.T, a.K, a.latent, a.tau_A, a.tau_B, a.x0[1], a.x0[2], a.x0[3]) > 0 && a.beta0 > 1 || error("Invalid positive parameter")
        all(x -> 0 <= x <= 1, a.x0[4:5]) && all(x -> 0 < x <= 1, a.umax) && all(0 .<= collect(a.initial) .<= collect(a.umax)) || error("Invalid phenotype/target")
        for (w, s) in ((a.w_A, a.scales_A), (a.w_B, a.scales_B))
            isapprox(sum(w), 1; atol = 1.0e-12) && all(>(0), s) || error("Invalid weights or scales")
        end
        for (floor, x0) in ((a.epsilon_A, a.x0[1]), (a.epsilon_B, a.x0[3]))
            floor === nothing || 0 < floor <= x0 || error("Density floor must be positive and initially feasible")
        end
        return
    end

    function validate_trajectory(x, a)
        minimum(x[1:3, :]) > 0 || error("Nonpositive density")
        return minimum(x[4:5, :]) >= -1.0e-8 && maximum(x[4:5, :]) <= 1 + 1.0e-8 || error("Phenotype bounds violated")
    end

    function game(a = Parameters())
        validate_parameters(a)
        return Game(;
            name = "Phage",
            state_labels = ["x₁: susceptible bacteria", "x₂: infected bacteria", "x₃: free phage", "x₄: resistance", "x₅: infectivity"],
            control_labels = ["u₁: target resistance", "u₂: target infectivity"], owners = [1, 2],
            dynamics! = (dx, x, u, t) -> dynamics!(dx, x, u, t, a),
            objective = (x, i) -> objective(x, i, a),
            constraints = constraints(a),
            x0 = vcat(log.(collect(a.x0[1:3])), collect(a.x0[4:5]), zeros(4)), T = a.T, quadratures = collect(6:9),
            lower = zeros(2), upper = collect(a.umax), initial = collect(a.initial),
            state_lower = [-Inf, -Inf, -Inf, 0, 0, -Inf, -Inf, -Inf, -Inf],
            state_upper = [Inf, Inf, Inf, 1, 1, Inf, Inf, Inf, Inf],
            running_objectives = [6, 7],
            decode = x -> vcat(exp.(x[1:3]), x[4:end]),
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
    CorleoneGame.run_logged(Phage.main, @__DIR__)
end
