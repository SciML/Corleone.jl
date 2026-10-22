# Category: episodical, cellular, mutualism
# One batch consumes substrate; harvest and refilling are outside the model.
# Minimum-time game on normalized s in [0,1]; biological model and choices are local.
using CorleoneGame
module BatchFermentation
    using CorleoneGame

    Base.@kwdef struct Parameters
        x0::NTuple{6, Float64} = (0.4, 0.35, 0.3, 5.0, 0.1, 0.0)
        target::Float64 = 0.8
        growth::NTuple{3, Float64} = (0.7, 0.6, 0.5)
        saturation::Float64 = 0.5
        maintenance::Float64 = 0.03
        production::NTuple{3, Float64} = (0.8, 0.7, 0.6)
        yield::Float64 = 0.7
        intermediate_decay::Float64 = 0.05
        duration::Float64 = 5.0
        duration_bounds::NTuple{2, Float64} = (0.1, 20.0)
        initial::NTuple{3, Float64} = (0.5, 0.45, 0.4)
        weights::Matrix{Float64} = [0.75 0.15 0.1; 0.55 0.3 0.15; 0.45 0.25 0.3]
        budgets::NTuple{3, Union{Nothing, Float64}} = (2.5, 2.0, 1.5)
        path_limits::NTuple{3, Float64} = (1.5, 0.2, 0.15)
    end

    function algorithm_options(; kwargs...)
        defaults = (; shooting = :single, nlp_solver = :ipopt, damping = 1.0, max_rounds = 300, extra_starts = Float64[], solver_options = (; hessian_approximation = "exact"))
        return Options(; merge(defaults, (; kwargs...))...)
    end

    function physical!(dx, x, u, s, a)
        S, M = x[4], x[5]
        uptake = S / (a.saturation + S)
        r1 = a.production[1] * u[1] * x[1] * uptake
        r2 = a.production[2] * u[2] * x[2] * M / (a.saturation + M)
        r3 = a.production[3] * u[3] * x[3] * M / (a.saturation + M)
        for i in 1:3
            dx[i] = (a.growth[i] * (1 - 0.5 * u[i]) * uptake - a.maintenance) * x[i]
        end
        dx[4] = -sum(a.growth[i] * (1 - 0.5 * u[i]) * uptake * x[i] / a.yield for i in 1:3) - r1
        dx[5] = r1 - r2 - r3 - a.intermediate_decay * M
        dx[6] = r2 + r3
        return nothing
    end
    burden(x, u, i, a) = x[5]^2 - (0.1 * i) * x[i]


    # p contains three biological controls followed by the constant duration parameter.
    function dynamics!(dx, x, p, s, a)
        u = view(p, 1:3); duration = p[4]
        physical!(view(dx, 1:6), view(x, 1:6), u, s, a)
        dx[1:6] .*= duration
        for i in 1:3
            dx[6 + i] = duration * u[i]^2
            dx[9 + i] = duration * burden(x, u, i, a)
        end
        return nothing
    end

    objective(x, i, p, a) = a.weights[i, 1] * p[1] + a.weights[i, 2] * x[6 + i] + a.weights[i, 3] * x[9 + i]

    function constraints(a)
        cs = [StateConstraint(; player = i, index = 6, label = "product quota", lower = a.target, upper = a.target) for i in 1:3]
        for i in 1:3
            a.budgets[i] === nothing || push!(
                cs, StateConstraint(;
                    player = i, index = 6 + i,
                    label = "squared budget u$(subscript(i))", upper = a.budgets[i]
                )
            )
        end
        push!(cs, StateConstraint(; player = 1, index = 5, label = "ceiling x$(subscript(5))", upper = a.path_limits[1], path = true))
        push!(cs, StateConstraint(; player = 2, index = 2, label = "floor x$(subscript(2))", lower = a.path_limits[2], path = true))
        push!(cs, StateConstraint(; player = 3, index = 3, label = "floor x$(subscript(3))", lower = a.path_limits[3], path = true))
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
        a.target > a.x0[6] || error("Target must prescribe a nonzero transfer in the intended direction")
        all(isfinite, a.x0) && isfinite(a.target) || error("Nonfinite state or target")
        0 < a.duration_bounds[1] <= a.duration <= a.duration_bounds[2] < Inf || error("Invalid duration bounds")
        size(a.weights) == (3, 3) && all(>=(0), a.weights) && all(isapprox.(vec(sum(a.weights; dims = 2)), 1.0; atol = 1.0e-12)) || error("Invalid objective weights")
        all(b -> b === nothing || (isfinite(b) && b >= 0), a.budgets) || error("Invalid budgets")
        return Game(;
            name = "BatchFermentation", state_labels = ["X₁: upstream biomass", "X₂: producer biomass B", "X₃: producer biomass C", "S: substrate", "M: intermediate", "P: product"], control_labels = ["u₁: intermediate production", "u₂: product synthesis", "u₃: product synthesis"], owners = [1, 2, 3],
            dynamics! = (dx, x, p, s) -> dynamics!(dx, x, p, s, a), objective = (x, i, p) -> objective(x, i, p, a),
            constraints = constraints(a), x0 = vcat(collect(a.x0), zeros(6)), T = 1.0, quadratures = collect(7:12),
            lower = zeros(3), upper = ones(3), initial = collect(a.initial), running_objectives = collect(7:12),
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
    CorleoneGame.run_logged(BatchFermentation.main, @__DIR__)
end
