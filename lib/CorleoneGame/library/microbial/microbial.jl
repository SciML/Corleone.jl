# Category: periodical, cellular, mutualism, variablePlayers
# A maintained cross-feeding chemostat under recurring feed; all stocks return.
# N-player microbial cross-feeding case.
# Run from lib/CorleoneGame with: julia --project=. library/microbial/microbial.jl [single|multiple|synthetic]
using CorleoneGame

module Microbial
    using LinearAlgebra
    using CorleoneGame

    # Return band 1e-2 (absolute): with 1e-4 the 21 shared return conditions pin every
    # strain's secretion to the calibrated reference allocation (tested 2026-09-26).
    # Synthetic periodic extension: fixed phase-zero data were prepared with constant
    # reference controls (tight cycle integration).
    # Custom kinetics, periods, or reference controls require new compatible fixed x0.
    # Approximate closure is not a claim of long-run orbital stability or equilibrium.
    struct Parameters
        forcing_period::Float64; forcing_amplitude::Float64; periodicity_tolerance::Float64
        N::Int; T::Float64; B0::Float64; Kdef::Float64; epsilon_M::Float64
        mu_max::Vector{Float64}; K_S::Vector{Float64}; K_M::Matrix{Float64}
        c::Vector{Float64}; alpha::Vector{Float64}; Y_S::Vector{Float64}; Y_M::Matrix{Float64}
        D::Float64; S_in::Float64; x0::Vector{Float64}; weights::Matrix{Float64}
        adjacency::BitMatrix; initial_control::Vector{Float64}
    end

    function Parameters(;
            N = 10, forcing_period = 80.0, T = forcing_period, forcing_amplitude = 0.9, periodicity_tolerance = 1.0e-2, B0 = 2.0, Kdef = 0.35, epsilon_M = 1.0e-5,
            mu_max = nothing, K_S = nothing, K_M = nothing, c = nothing, alpha = nothing,
            Y_S = nothing, Y_M = nothing, D = 0.05, S_in = 10.0, x0 = nothing,
            weights = nothing, adjacency = nothing, initial_control = nothing
        )
        N >= 2 || error("N must be at least 2")
        mumax = mu_max === nothing ? (N == 2 ? [0.55, 0.5] : [0.45 + 0.025 * i for i in 1:N]) : collect(Float64, mu_max)
        # Half-saturation near the substrate level makes growth respond to the feed cycle.
        ks = K_S === nothing ? (N == 2 ? [0.8, 1.0] : [5.0 + 0.7 * i for i in 1:N]) : collect(Float64, K_S)
        km = K_M === nothing ? (N == 2 ? [0.12 0.12; 0.1 0.1] : [0.06 + 0.004 * mod(i + 2j, 7) for i in 1:N, j in 1:N]) : Matrix{Float64}(K_M)
        cc = c === nothing ? (N == 2 ? [0.35, 0.3] : [0.15 + 0.02 * i for i in 1:N]) : collect(Float64, c)
        aa = alpha === nothing ? (N == 2 ? [0.55, 0.5] : [0.45 + 0.04 * i for i in 1:N]) : collect(Float64, alpha)
        ys = Y_S === nothing ? (N == 2 ? [0.65, 0.6] : [0.5 + 0.015 * i for i in 1:N]) : collect(Float64, Y_S)
        ym = Y_M === nothing ? fill(8.0, N, N) : Matrix{Float64}(Y_M)
        state0 = x0 === nothing ? (N == 2 ? [0.15, 0.12, 6.0, 0.02, 0.02] : vcat([0.08 + 0.02 * i for i in 1:N], 6.0, [0.025 + 0.008 * mod(i, 5) for i in 1:N])) : collect(Float64, x0)
        # Calibrated coexistence at fixed positive strain densities for the default N=10.
        # Other network sizes retain configurable seeds and need their own cycle preparation.
        if N == 10
            mu_max === nothing && (mumax = [0.13906911751595225, 0.15349715538425818, 0.1506718399047913, 0.16220497062596426, 0.16480608061340293, 0.1752300865080435, 0.18040941547175168, 0.1903919219621705, 0.19706249108844417, 0.20691312921825342])
            x0 === nothing && (state0 = [0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22000000000000003, 0.24, 0.26, 0.28, 2.7166722580624003, 0.4602340171668921, 0.610507391814178, 0.7733384874174605, 0.9605044770830007, 1.1595419135663931, 1.3835755359295316, 1.6188845304979416, 1.8797601246365985, 2.151413227537054, 2.449110335935321])
        end
        ww = weights === nothing ? (N == 2 ? [0.58 0.22 0.05 0.15; 0.62 0.25 0.05 0.08] : [k == 1 ? 0.4 : k == 2 ? 0.01 + 0.02 * (i - 1) / (N - 1) : k == 3 ? 0.08 : 0.51 - 0.02 * (i - 1) / (N - 1) for i in 1:N, k in 1:4]) : Matrix{Float64}(weights)
        graph = adjacency === nothing ? falses(N, N) : BitMatrix(adjacency)
        # Reciprocal partners for N>2; all pairs still compete for shared substrate.
        if adjacency === nothing
            for i in 1:N
                partner = N == 2 ? 3 - i : isodd(i) ? (i == N ? N - 1 : i + 1) : i - 1
                graph[i, partner] = true
            end
        end
        u0 = initial_control === nothing ? fill(0.35, N) : collect(Float64, initial_control)
        return Parameters(Float64(forcing_period), Float64(forcing_amplitude), Float64(periodicity_tolerance), N, Float64(T), Float64(B0), Float64(Kdef), Float64(epsilon_M), mumax, ks, km, cc, aa, ys, ym, Float64(D), Float64(S_in), state0, ww, graph, u0)
    end

    "Case-specific algorithm defaults; keyword arguments override these settings."
    function algorithm_options(; kwargs...)
        defaults = (; shooting = :single, nlp_solver = :ipopt, max_rounds = 20, damping = 0.65)
        return Options(; merge(defaults, (; kwargs...))...)
    end

    player_label(i) = i <= 26 ? string(Char(64 + i)) : "Player $i"

    quadrature_index(p, i, k) = 2p.N + 2 + 4(i - 1) + k - 1
    incoming(p, i) = findall(@view p.adjacency[i, :])

    function dynamics!(dx, x, u, t, p)
        N = p.N
        X = @view x[1:N]; S = max(x[N + 1], 0.0); M = @view x[(N + 2):(2N + 1)]; growth = similar(X)
        for i in 1:N
            nutrient = S / (p.K_S[i] + S)
            for j in incoming(p, i)
                mj = max(M[j], 0.0); nutrient *= mj / (p.K_M[i, j] + mj)
            end
            growth[i] = p.mu_max[i] * (1 - p.c[i] * u[i]) * nutrient
            dx[i] = (growth[i] - p.D) * X[i]
        end
        dx[N + 1] = p.D * (p.S_in * (1 + p.forcing_amplitude * sin(2pi * t / p.forcing_period)) - S) - sum(growth[i] * X[i] / p.Y_S[i] for i in 1:N)
        for j in 1:N
            consumption = sum(((M[j] > 0 ? growth[i] * X[i] / p.Y_M[i, j] : 0.0) for i in 1:N if p.adjacency[i, j]); init = zero(M[j]))
            dx[N + 1 + j] = p.alpha[j] * u[j] * X[j] - p.D * max(M[j], 0.0) - consumption
        end
        for i in 1:N
            js = incoming(p, i)
            dx[quadrature_index(p, i, 1)] = -X[i] / (p.B0 * p.T)
            dx[quadrature_index(p, i, 2)] = (u[i]^2 + p.epsilon_M * M[i]^2) / p.T
            dx[quadrature_index(p, i, 3)] = sum(p.Kdef / (p.Kdef + max(M[j], 0.0)) for j in js) / (length(js) * p.T)
            dx[quadrature_index(p, i, 4)] = -sum(X[j] / p.B0 for j in js) / (length(js) * p.T)
        end
        return
    end

    function objective(x, player, p)
        return dot(
            @view(p.weights[player, :]),
            @view(x[quadrature_index(p, player, 1):quadrature_index(p, player, 4)])
        )
    end

    # All strains, shared substrate, and exchanged metabolites close for every player.
    constraints(p) = periodic_constraints(p.x0, p.N; tolerance = p.periodicity_tolerance)

    function validate_parameters(p)
        validate_periodic_clock(p.T, p.forcing_period, p.forcing_amplitude)
        N = p.N
        length(p.mu_max) == N && length(p.K_S) == N && length(p.c) == N && length(p.alpha) == N && length(p.Y_S) == N || error("Strain parameter dimensions disagree")
        size(p.K_M) == (N, N) && size(p.Y_M) == (N, N) && size(p.weights) == (N, 4) || error("Pairwise parameter dimensions disagree")
        length(p.x0) == 2N + 1 && length(p.initial_control) == N || error("Initial-state dimensions disagree")
        all(p.adjacency[i, i] == false for i in 1:N) || error("A strain cannot require its own metabolite")
        all(sum(p.adjacency[i, :]) > 0 for i in 1:N) || error("Every strain must require a metabolite")
        all(isfinite, p.x0) && all(p.x0 .>= 0) || error("Initial states must be nonnegative")
        all(isfinite, p.weights) && all(abs.(sum(p.weights, dims = 2) .- 1) .< 1.0e-12) || error("Weights must sum to one")
        all(0.02 .<= p.initial_control .<= 0.95) || error("Initial controls are outside bounds")
        p.T > 0 && p.B0 > 0 && p.Kdef > 0 && p.epsilon_M >= 0 && p.D > 0 && p.S_in > 0 || error("Invalid scalar parameter")
        return all(p.mu_max .> 0) && all(p.K_S .> 0) && all(p.K_M .> 0) && all(p.c .>= 0) && all(p.alpha .> 0) && all(p.Y_S .> 0) && all(p.Y_M .> 0) || error("Invalid kinetic parameter")
    end

    function validate_trajectory(x, p)
        return minimum(x[1:(2p.N + 1), :]) >= -1.0e-7 || error("Negative microbial state")
    end

    function game(p = Parameters())
        validate_parameters(p); N = p.N; q0 = quadrature_index(p, 1, 1)
        labels = vcat(["X$(subscript(i))" for i in 1:N], ["SCALE 0.1 S"], ["M$(subscript(i))" for i in 1:N])
        controls = ["u$(subscript(i)): secretion allocation" for i in 1:N]
        physical = 2N + 1
        return Game(;
            name = "Microbial cross-feeding N=$N",
            state_labels = labels,
            control_labels = controls, owners = collect(1:N),
            dynamics! = (dx, x, u, t) -> dynamics!(dx, x, u, t, p),
            objective = (x, i) -> objective(x, i, p),
            constraints = constraints(p),
            x0 = vcat(p.x0, zeros(4N)), T = p.T, quadratures = collect(q0:(q0 + 4N - 1)),
            lower = fill(0.02, N), upper = fill(0.95, N), initial = p.initial_control,
            state_lower = vcat(fill(0.0, physical), fill(-Inf, 4N)),
            state_upper = fill(Inf, physical + 4N),
            gain_scales = ones(N), running_objectives = collect(q0:(q0 + 4N - 1)),
            validate = x -> validate_trajectory(x, p),
            metadata = p
        )
    end

    function main(; shooting = :single, parameters = Parameters(), options = algorithm_options(; shooting), folder = joinpath(@__DIR__, options.shooting == :single ? "output" : "output_multiple"), synthetic = false)
        g = game(parameters)
        synthetic && return create_synthetic_data(g, folder)
        return run_game(g; options, folder)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    CorleoneGame.run_logged(Microbial.main, @__DIR__)
end
