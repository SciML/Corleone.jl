# Category: episodical, cellular, antagonism
# One acute infection proceeds toward clearance, not back to the initial infection.
# Host-virus game on normalized s in [0,1]; biological model and choices are local.
using CorleoneGame
module ImmuneClearance
    using CorleoneGame

    Base.@kwdef struct Parameters
        x0::NTuple{6, Float64} = (1.0, 0.2, 2.0, 0.1, 0.1, 0.0)
        target::Float64 = 0.2
        beta::Float64 = 0.25
        infected_decay::Float64 = 0.6
        production::Float64 = 0.7
        clearance::Float64 = 0.8
        killing::Float64 = 0.6
        activation::NTuple{2, Float64} = (0.8, 0.6)
        effector_decay::NTuple{2, Float64} = (0.5, 0.35)
        damage::NTuple{2, Float64} = (0.2, 0.15)
        recovery::Float64 = 0.4
        # Viral immune evasion: fraction of cytotoxic killing and of antibody neutralization
        # avoided at full investment, and the fraction of virion production lost at full investment.
        evasion::Float64 = 0.8
        evasion_cost::Float64 = 0.3
        duration::Float64 = 5.0
        duration_bounds::NTuple{2, Float64} = (0.1, 20.0)
        initial::NTuple{3, Float64} = (0.5, 0.45, 0.3)
        # Host: (duration, squared activation effort, viral and damage burden).
        w_A::NTuple{3, Float64} = (0.5, 0.3, 0.2)
        # Virus: (transmission proxy, squared evasion effort).
        w_B::NTuple{2, Float64} = (0.6, 0.4)
        budgets::NTuple{2, Union{Nothing, Float64}} = (2.0, 1.0)
        # Host restrictions: susceptible-cell floor and tissue-damage ceiling.
        path_limits::NTuple{2, Float64} = (0.5, 0.8)
    end

    function algorithm_options(; kwargs...)
        defaults = (; shooting = :single, nlp_solver = :ipopt, damping = 1.0, max_rounds = 300, extra_starts = Float64[], solver_options = (; hessian_approximation = "exact"))
        return Options(; merge(defaults, (; kwargs...))...)
    end

    function physical!(dx, x, u, s, a)
        H, I, V, E, Ab, Z = x
        infection = a.beta * H * V / (1 + V)
        dx[1] = -infection
        dx[2] = infection - (a.infected_decay + a.killing * (1 - a.evasion * u[3]) * E) * I
        dx[3] = a.production * (1 - a.evasion_cost * u[3]) * I - (a.clearance + (1 - a.evasion * u[3]) * Ab) * V
        dx[4] = a.activation[1] * u[1] * V / (1 + V) - a.effector_decay[1] * E
        dx[5] = a.activation[2] * u[2] * V / (1 + V) - a.effector_decay[2] * Ab
        dx[6] = a.damage[1] * E^2 + a.damage[2] * I - a.recovery * Z
        return nothing
    end

    # p contains three biological controls (host, host, virus) followed by the duration.
    function dynamics!(dx, x, p, s, a)
        u = view(p, 1:3); duration = p[4]
        physical!(view(dx, 1:6), view(x, 1:6), u, s, a)
        dx[1:6] .*= duration
        dx[7] = duration * (u[1]^2 + u[2]^2)
        dx[8] = duration * u[3]^2
        dx[9] = duration * (x[3]^2 + x[6]^2)
        dx[10] = duration * x[3]
        return nothing
    end

    objective(x, i, p, a) = i == 1 ? a.w_A[1] * p[1] + a.w_A[2] * x[7] + a.w_A[3] * x[9] :
        -a.w_B[1] * x[10] + a.w_B[2] * x[8]

    function constraints(a)
        # The clearance target is the host's restriction; the virus is not bound by it.
        cs = [StateConstraint(; player = 1, index = 3, label = "viral-load target", lower = a.target, upper = a.target)]
        for i in 1:2
            a.budgets[i] === nothing || push!(
                cs, StateConstraint(;
                    player = i, index = 6 + i,
                    label = i == 1 ? "activation budget" : "evasion budget", upper = a.budgets[i]
                )
            )
        end
        push!(cs, StateConstraint(; player = 1, index = 1, label = "floor x$(subscript(1))", lower = a.path_limits[1], path = true))
        push!(cs, StateConstraint(; player = 1, index = 6, label = "ceiling x$(subscript(6))", upper = a.path_limits[2], path = true))
        # Number visible bounds in plot order; shared copies retain one label.
        numbers = Dict{Tuple{Int, Float64, Float64, Bool}, Int}()
        return map(cs) do c
            startswith(c.label, "!") && return c
            key = (c.index, c.lower, c.upper, c.path)
            i = get!(numbers, key, length(numbers) + 1)
            StateConstraint(;
                player = c.player, index = c.index, label = "g$(subscript(i)): $(c.label)",
                lower = c.lower, upper = c.upper, path = c.path
            )
        end
    end

    function game(a = Parameters())
        0 < a.target < a.x0[3] || error("Target must prescribe a nonzero transfer in the intended direction")
        all(isfinite, a.x0) && isfinite(a.target) || error("Nonfinite state or target")
        0 < a.duration_bounds[1] <= a.duration <= a.duration_bounds[2] < Inf || error("Invalid duration bounds")
        for w in (a.w_A, a.w_B)
            all(>=(0), w) && isapprox(sum(w), 1.0; atol = 1.0e-12) || error("Invalid objective weights")
        end
        0 <= a.evasion <= 1 && 0 <= a.evasion_cost < 1 || error("Evasion efficacy and cost must be fractions")
        all(b -> b === nothing || (isfinite(b) && b >= 0), a.budgets) || error("Invalid budgets")
        return Game(;
            name = "ImmuneClearance", state_labels = ["H: susceptible cells", "I: infected cells", "V: viral load", "E: cytotoxic effectors", "Ab: antibody activity", "Z: tissue damage"],
            control_labels = ["u₁: cytotoxic activation", "u₂: antibody activation", "u₃: immune evasion"], owners = [1, 1, 2],
            dynamics! = (dx, x, p, s) -> dynamics!(dx, x, p, s, a), objective = (x, i, p) -> objective(x, i, p, a),
            constraints = constraints(a), x0 = vcat(collect(a.x0), zeros(4)), T = 1.0, quadratures = collect(7:10),
            lower = zeros(3), upper = ones(3), initial = collect(a.initial), running_objectives = collect(7:10),
            parameters = [a.duration], parameter_owners = [1], parameter_labels = ["duration"],
            bounds_p = ([a.duration_bounds[1]], [a.duration_bounds[2]]), time_parameter = 1, metadata = a
        )
    end

    # Feasible constant-control initial duration; each case can replace this initializer.
    function initial_parameters(g; n = 20)
        u = initial_controls(g, n); c = first(g.constraints)
        residual(d) = simulate(g, u; samples = 1, parameters = [d]).x[c.index, end] - c.lower
        lo, hi = g.bounds_p[1][1], g.bounds_p[2][1]; flo = residual(lo)
        flo * residual(hi) <= 0 || error("Initial controls do not bracket the target within duration_bounds")
        for k in 1:40
            mid = (lo + hi) / 2; fm = residual(mid)
            if flo * fm <= 0
                hi = mid
            else
                lo = mid; flo = fm
            end
        end
        return [(lo + hi) / 2]
    end

    function main(;
            shooting = :single, parameters = Parameters(), options = algorithm_options(; shooting),
            folder = joinpath(@__DIR__, options.shooting == :single ? "output" : "output_multiple"), synthetic = false
        )
        synthetic && return create_synthetic_data(game(parameters), folder)
        g = game(parameters)
        return run_game(g; options, folder, parameters = initial_parameters(g; n = options.intervals))
    end
end
if abspath(PROGRAM_FILE) == @__FILE__
    CorleoneGame.run_logged(ImmuneClearance.main, @__DIR__)
end
