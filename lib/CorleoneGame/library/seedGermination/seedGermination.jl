# Category: seasonal, population, competition
# A cohort progresses toward emergence; reproduction and dormancy are omitted.
# Minimum-time game on normalized s in [0,1]; biological model and choices are local.
using CorleoneGame
module SeedGermination
    using CorleoneGame

    Base.@kwdef struct Parameters
        x0::NTuple{7, Float64} = (1.5, 0.15, 0.12, 0.1, 0.0, 0.0, 0.0)
        target::Float64 = 1.0
        water_supply::Float64 = 0.4
        water_capacity::Float64 = 2.0
        water_use::Float64 = 0.2
        activation::NTuple{3, Float64} = (1.0, 0.85, 0.7)
        decay::NTuple{3, Float64} = (0.4, 0.35, 0.3)
        growth::NTuple{3, Float64} = (0.5, 0.6, 0.7)
        saturation::Float64 = 0.4
        duration::Float64 = 5.0
        duration_bounds::NTuple{2, Float64} = (0.1, 20.0)
        initial::NTuple{3, Float64} = (0.5, 0.45, 0.4)
        weights::Matrix{Float64} = [0.75 0.15 0.1; 0.55 0.3 0.15; 0.45 0.25 0.3]
        budgets::NTuple{3, Union{Nothing, Float64}} = (3.0, 2.2, 1.5)
        path_limits::NTuple{3, Float64} = (0.5, 2.4, 2.8)
    end

    function algorithm_options(; kwargs...)
        defaults = (; shooting = :single, nlp_solver = :ipopt, damping = 1.0, max_rounds = 300, extra_starts = Float64[])
        return Options(; merge(defaults, (; kwargs...))...)
    end

    function physical!(dx, x, u, s, a)
        W = x[1]
        dx[1] = a.water_supply * (a.water_capacity - W) - a.water_use * W * sum(u)
        for i in 1:3
            dx[1 + i] = a.activation[i] * u[i] * W / (a.saturation + W) - a.decay[i] * x[1 + i]
            dx[4 + i] = a.growth[i] * x[1 + i] * W / (a.saturation + W)
        end
        return nothing
    end
    burden(x, u, i, a) = (x[4 + i] - sum(x[5:7]) / 3)^2


    # p contains three biological controls followed by the constant duration parameter.
    function dynamics!(dx, x, p, s, a)
        u = view(p, 1:3); duration = p[4]
        physical!(view(dx, 1:7), view(x, 1:7), u, s, a)
        dx[1:7] .*= duration
        for i in 1:3
            dx[7 + i] = duration * u[i]^2
            dx[10 + i] = duration * burden(x, u, i, a)
        end
        return nothing
    end

    objective(x, i, p, a) = a.weights[i, 1] * p[1] + a.weights[i, 2] * x[7 + i] + a.weights[i, 3] * x[10 + i]

    function constraints(a)
        cs = [StateConstraint(; player = i, index = 5, label = "leader radicle emergence", lower = a.target, upper = a.target) for i in 1:3]
        for i in 1:3
            a.budgets[i] === nothing || push!(
                cs, StateConstraint(;
                    player = i, index = 7 + i,
                    label = "squared budget u$(subscript(i))", upper = a.budgets[i]
                )
            )
        end
        push!(cs, StateConstraint(; player = 1, index = 1, label = "floor x$(subscript(1))", lower = a.path_limits[1], path = true))
        push!(cs, StateConstraint(; player = 2, index = 6, label = "ceiling x$(subscript(6))", upper = a.path_limits[2], path = true))
        push!(cs, StateConstraint(; player = 3, index = 7, label = "ceiling x$(subscript(7))", upper = a.path_limits[3], path = true))
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
        a.target > a.x0[5] || error("Target must prescribe a nonzero transfer in the intended direction")
        all(isfinite, a.x0) && isfinite(a.target) || error("Nonfinite state or target")
        0 < a.duration_bounds[1] <= a.duration <= a.duration_bounds[2] < Inf || error("Invalid duration bounds")
        size(a.weights) == (3, 3) && all(>=(0), a.weights) && all(isapprox.(vec(sum(a.weights; dims = 2)), 1.0; atol = 1.0e-12)) || error("Invalid objective weights")
        all(b -> b === nothing || (isfinite(b) && b >= 0), a.budgets) || error("Invalid budgets")
        return Game(;
            name = "SeedGermination", state_labels = ["W: shared water", "A₁: embryo activation A", "A₂: embryo activation B", "A₃: embryo activation C", "G₁: radicle progress A", "G₂: radicle progress B", "G₃: radicle progress C"], control_labels = ["u₁: embryo activation", "u₂: embryo activation", "u₃: embryo activation"], owners = [1, 2, 3],
            dynamics! = (dx, x, p, s) -> dynamics!(dx, x, p, s, a), objective = (x, i, p) -> objective(x, i, p, a),
            constraints = constraints(a), x0 = vcat(collect(a.x0), zeros(6)), T = 1.0, quadratures = collect(8:13),
            lower = zeros(3), upper = ones(3), initial = collect(a.initial), running_objectives = collect(8:13),
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
    CorleoneGame.run_logged(SeedGermination.main, @__DIR__)
end
