# Category: periodical, molecular, regulation
# An intrinsic clock cycle closes all ten concentrations; duration belongs to A.
# Slaby et al. (2007), Appendix B: ten-state Drosophila PER/TIM oscillator.
# The two-player interventions, objectives, and constraints are benchmark additions.
using CorleoneGame
module CircadianCycle
    using CorleoneGame

    Base.@kwdef struct Parameters
        x0::NTuple{10, Float64} = (0.4884645232074144, 0.19727176556891227, 0.2725715265391962, 0.40257808568446946, 0.4884645232074144, 0.19727176556891227, 0.2725715265391962, 0.40257808568446946, 0.7156565510581766, 2.0447330030232895)
        vs::NTuple{2, Float64} = (1.0, 1.0)
        vm::NTuple{2, Float64} = (0.7, 0.7)
        Km::NTuple{2, Float64} = (0.2, 0.2)
        ks::NTuple{2, Float64} = (0.9, 0.9)
        vd::NTuple{2, Float64} = (2.0, 2.0)
        KI::NTuple{2, Float64} = (1.0, 1.0)
        Kd::NTuple{2, Float64} = (0.2, 0.2)
        V::NTuple{4, Float64} = (8.0, 1.0, 8.0, 1.0)
        K::NTuple{4, Float64} = (2.0, 2.0, 2.0, 2.0)
        k1::Float64 = 0.6
        k2::Float64 = 0.2
        k3::Float64 = 1.2
        k4::Float64 = 0.6
        kd::Float64 = 0.01
        kdC::Float64 = 0.01
        kdN::Float64 = 0.01
        hill::Int = 4
        transcription_gain::Float64 = 0.6
        light_gain::Float64 = 2.0
        # A owns the intrinsic period. The objective is total per-cycle cost
        # (duration plus accumulated effort/burden), not a long-run average rate.
        # Budgets are fixed per cycle; the positive lower bound excludes zero duration.
        # Fixed x0 comes from a nonstationary reference orbit, not a steady state.
        duration::Float64 = 24.13458673765308
        duration_bounds::NTuple{2, Float64} = (12.0, 36.0)
        periodicity_tolerance::Float64 = 1.0e-4 # nM; zero requests exact periodic closure.
        initial::NTuple{2, Float64} = (0.5, 0.5)
        weights::Matrix{Float64} = [0.65 0.2 0.15; 0.45 0.35 0.2]
        budgets::NTuple{2, Union{Nothing, Float64}} = (12.0, 10.0)
        path_limits::NTuple{2, Float64} = (5.0, 4.0)
    end

    function algorithm_options(; kwargs...)
        defaults = (;
            shooting = :single, nlp_solver = :ipopt, damping = 1.0, max_rounds = 300, max_iters = 500,
            extra_starts = Float64[], solver_options = (; hessian_approximation = "exact"),
        )
        return Options(; merge(defaults, (; kwargs...))...)
    end

    # Order: MP,P0,P1,P2,MT,T0,T1,T2,C,CN. Concentrations nM; physical time hours.
    function physical!(dx, x, u, s, a)
        C, CN = x[9], x[10]
        association = a.k3 * x[4] * x[8] - a.k4 * C
        for j in 1:2
            k = 4 * (j - 1); M, X0, X1, X2 = x[k + 1], x[k + 2], x[k + 3], x[k + 4]
            v1 = a.V[1] * X0 / (a.K[1] + X0); v2 = a.V[2] * X1 / (a.K[2] + X1)
            v3 = a.V[3] * X1 / (a.K[3] + X1); v4 = a.V[4] * X2 / (a.K[4] + X2)
            synthesis = a.vs[j] * (j == 1 ? 1 + a.transcription_gain * (u[1] - 0.5) : 1)
            degradation = a.vd[j] + (j == 2 ? a.light_gain * (u[2] - 0.5) : 0)
            dx[k + 1] = synthesis * a.KI[j]^a.hill / (a.KI[j]^a.hill + CN^a.hill) - a.vm[j] * M / (a.Km[j] + M) - a.kd * M
            dx[k + 2] = a.ks[j] * M - v1 + v2 - a.kd * X0
            dx[k + 3] = v1 - v2 - v3 + v4 - a.kd * X1
            dx[k + 4] = v3 - v4 - association - degradation * X2 / (a.Kd[j] + X2) - a.kd * X2
        end
        dx[9] = association - a.k1 * C + a.k2 * CN - a.kdC * C
        dx[10] = a.k1 * C - a.k2 * CN - a.kdN * CN
        return nothing
    end
    burden(x, u, i, a) = i == 1 ? (x[1] - x[5])^2 : (sum(x[2:4]) - sum(x[6:8]))^2

    function dynamics!(dx, x, p, s, a)
        u = view(p, 1:2); duration = p[3]
        physical!(view(dx, 1:10), view(x, 1:10), u, s, a)
        dx[1:10] .*= duration
        for i in 1:2
            dx[10 + i] = duration * u[i]^2
            dx[12 + i] = duration * burden(x, u, i, a)
        end
        return nothing
    end
    objective(x, i, p, a) = a.weights[i, 1] * p[1] + a.weights[i, 2] * x[10 + i] + a.weights[i, 3] * x[12 + i]
    function constraints(a)
        cs = StateConstraint[]
        for i in 1:2, j in 1:10
            push!(
                cs, StateConstraint(;
                    player = i, index = j, label = "!floor x$(subscript(j))",
                    lower = a.x0[j] - a.periodicity_tolerance
                )
            )
            push!(
                cs, StateConstraint(;
                    player = i, index = j, label = "!g$(subscript(j)): ceiling x$(subscript(j))",
                    upper = a.x0[j] + a.periodicity_tolerance
                )
            )
        end
        for i in 1:2
            a.budgets[i] === nothing || push!(cs, StateConstraint(; player = i, index = 10 + i, label = "squared budget u$(subscript(i))", upper = a.budgets[i]))
        end
        push!(cs, StateConstraint(; player = 1, index = 1, label = "ceiling x$(subscript(1))", upper = a.path_limits[1], path = true))
        push!(cs, StateConstraint(; player = 2, index = 8, label = "ceiling x$(subscript(8))", upper = a.path_limits[2], path = true))
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
        isfinite(a.periodicity_tolerance) && 0 <= a.periodicity_tolerance < minimum(a.x0) || error("Invalid cycle return tolerance")
        all(>(0), a.x0) || error("Initial concentrations must be positive")
        0 < a.duration_bounds[1] <= a.duration <= a.duration_bounds[2] || error("A positive period lower bound is required")
        0 <= a.transcription_gain < 2 && 0 <= a.light_gain < 2a.vd[2] || error("Interventions must preserve positive kinetic rates")
        return Game(;
            name = "CircadianCycle", state_labels = ["MP: per mRNA", "P0: PER", "P1: PER-P", "P2: PER-PP", "MT: tim mRNA", "T0: TIM", "T1: TIM-P", "T2: TIM-PP", "C: cytosolic complex", "CN: nuclear complex"],
            control_labels = ["u₁: per transcription", "u₂: TIM degradation (light)"], owners = [1, 2],
            dynamics! = (dx, x, p, s) -> dynamics!(dx, x, p, s, a), objective = (x, i, p) -> objective(x, i, p, a),
            constraints = constraints(a), x0 = vcat(collect(a.x0), zeros(4)), T = 1.0, quadratures = collect(11:14),
            lower = zeros(2), upper = ones(2), initial = collect(a.initial), running_objectives = collect(11:14),
            parameters = [a.duration], parameter_owners = [1], parameter_labels = ["duration"],
            bounds_p = ([a.duration_bounds[1]], [a.duration_bounds[2]]), time_parameter = 1, metadata = a
        )
    end
    initial_parameters(g; n = 20) = copy(g.parameters)
    function main(;
            shooting = :single, parameters = Parameters(), options = algorithm_options(; shooting),
            folder = joinpath(@__DIR__, options.shooting == :single ? "output" : "output_multiple"), synthetic = false
        )
        synthetic && return create_synthetic_data(game(parameters), folder)
        return run_game(game(parameters); options, folder)
    end
end
if abspath(PROGRAM_FILE) == @__FILE__
    CorleoneGame.run_logged(CircadianCycle.main, @__DIR__)
end
