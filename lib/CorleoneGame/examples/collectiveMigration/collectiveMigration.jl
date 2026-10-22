# Category: episodical
# One migration leg ends elsewhere; return and offseason recovery are omitted.
# Minimum-time game on normalized s in [0,1]; biological model and choices are local.
using CorleoneGame
module CollectiveMigration
    using CorleoneGame

    Base.@kwdef struct Parameters
        x0::NTuple{9, Float64} = (0.0, -0.2, -0.4, 0.2, 0.2, 0.2, 1.0, 0.9, 0.8)
        target::Float64 = 4.0
        thrust::NTuple{3, Float64} = (1.2, 1.0, 0.9)
        drag::NTuple{3, Float64} = (0.5, 0.6, 0.7)
        alignment::Float64 = 0.2
        replenishment::Float64 = 0.15
        depletion::Float64 = 0.1
        energy_saturation::Float64 = 0.2
        duration::Float64 = 5.0
        duration_bounds::NTuple{2, Float64} = (0.1, 20.0)
        initial::NTuple{3, Float64} = (0.5, 0.45, 0.4)
        weights::Matrix{Float64} = [0.75 0.15 0.1; 0.55 0.3 0.15; 0.45 0.25 0.3]
        budgets::NTuple{3, Union{Nothing, Float64}} = (4.0, 3.0, 2.0)
        path_limits::NTuple{3, Float64} = (0.4, 0.35, 0.3)
    end

    function algorithm_options(; kwargs...)
        defaults = (; shooting = :single, nlp_solver = :ipopt, damping = 1.0, max_rounds = 300, extra_starts = Float64[])
        return Options(; merge(defaults, (; kwargs...))...)
    end

    function physical!(dx, x, u, s, a)
        meanv = sum(x[4:6]) / 3
        for i in 1:3
            v, E = x[3 + i], x[6 + i]
            dx[i] = v
            dx[3 + i] = a.thrust[i] * u[i] * E / (a.energy_saturation + E) - a.drag[i] * v^2 + a.alignment * (meanv - v)
            dx[6 + i] = a.replenishment * (1 - E) - a.depletion * u[i]^2 * E
        end
        return nothing
    end
    burden(x, u, i, a) = (x[i] - sum(x[1:3]) / 3)^2 + (1 - x[6 + i])^2


    # p contains three biological controls followed by the constant duration parameter.
    function dynamics!(dx, x, p, s, a)
        u = view(p, 1:3); duration = p[4]
        physical!(view(dx, 1:9), view(x, 1:9), u, s, a)
        dx[1:9] .*= duration
        for i in 1:3
            dx[9 + i] = duration * u[i]^2
            dx[12 + i] = duration * burden(x, u, i, a)
        end
        return nothing
    end

    objective(x, i, p, a) = a.weights[i, 1] * p[1] + a.weights[i, 2] * x[9 + i] + a.weights[i, 3] * x[12 + i]

    function constraints(a)
        cs = [StateConstraint(; player = i, index = 1, label = "leader destination", lower = a.target, upper = a.target) for i in 1:3]
        for i in 1:3
            a.budgets[i] === nothing || push!(
                cs, StateConstraint(;
                    player = i, index = 9 + i,
                    label = "squared budget u$(subscript(i))", upper = a.budgets[i]
                )
            )
        end
        push!(cs, StateConstraint(; player = 1, index = 7, label = "floor x$(subscript(7))", lower = a.path_limits[1], path = true))
        push!(cs, StateConstraint(; player = 2, index = 8, label = "floor x$(subscript(8))", lower = a.path_limits[2], path = true))
        push!(cs, StateConstraint(; player = 3, index = 9, label = "floor x$(subscript(9))", lower = a.path_limits[3], path = true))
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
        a.target > a.x0[1] || error("Target must prescribe a nonzero transfer in the intended direction")
        all(isfinite, a.x0) && isfinite(a.target) || error("Nonfinite state or target")
        0 < a.duration_bounds[1] <= a.duration <= a.duration_bounds[2] < Inf || error("Invalid duration bounds")
        size(a.weights) == (3, 3) && all(>=(0), a.weights) && all(isapprox.(vec(sum(a.weights; dims = 2)), 1.0; atol = 1.0e-12)) || error("Invalid objective weights")
        all(b -> b === nothing || (isfinite(b) && b >= 0), a.budgets) || error("Invalid budgets")
        return Game(;
            name = "CollectiveMigration", state_labels = ["q₁: position A", "q₂: position B", "q₃: position C", "v₁: speed A", "v₂: speed B", "v₃: speed C", "E₁: reserves A", "E₂: reserves B", "E₃: reserves C"], control_labels = ["u₁: propulsion", "u₂: propulsion", "u₃: propulsion"], owners = [1, 2, 3],
            dynamics! = (dx, x, p, s) -> dynamics!(dx, x, p, s, a), objective = (x, i, p) -> objective(x, i, p, a),
            constraints = constraints(a), x0 = vcat(collect(a.x0), zeros(6)), T = 1.0, quadratures = collect(10:15),
            lower = zeros(3), upper = ones(3), initial = collect(a.initial), running_objectives = collect(10:15),
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
    CorleoneGame.run_logged(CollectiveMigration.main, @__DIR__)
end
