# Game construction, validation, best responses, round iteration, point costs,
# periodic-return constraints, and file output on small analytic games.
using CorleoneGame, Test, Corleone

# Player A owns two controls, player B one; each player's cost is its own quadrature.
function analytic_game(; T = 1.0)
    function rhs!(dx, x, u, t)
        dx[1] = u[1] + 2u[2] - u[3]
        dx[2] = (u[1] - 0.2)^2 + (u[2] - 0.3)^2
        return dx[3] = (u[3] - 0.4)^2
    end
    return Game(;
        name = "Analytic",
        state_labels = ["x"],
        control_labels = ["A1", "A2", "B1"], owners = [1, 1, 2],
        dynamics! = rhs!,
        objective = (x, i) -> x[i + 1],
        x0 = zeros(3), T, quadratures = [2, 3],
        lower = zeros(3), upper = ones(3), initial = fill(0.7, 3)
    )
end

@testset "Validation and helper functions" begin
    g = analytic_game()
    @test player_count(g) == 2
    @test player_label(2, 1) == "A" && player_label(3, 3) == "C"
    @test subscript(12) == "₁₂"
    @test initial_controls(g, 4) == fill(0.7, 3, 4)
    CorleoneGame.validate(g, Options())
    @test_throws ErrorException CorleoneGame.validate(g, Options(shooting = :unknown))
    @test_throws ErrorException CorleoneGame.validate(g, Options(damping = 0.0))
    @test_throws ErrorException CorleoneGame.validate(g, Options(intervals = 8, refined_intervals = 4))
    # Players must be numbered consecutively; quadratures must start at zero.
    bad = Game(;
        name = "Bad owners", state_labels = ["x"], control_labels = ["A", "C"], owners = [1, 3],
        dynamics! = (dx, x, u, t) -> fill!(dx, 0), objective = (x, i) -> x[1],
        x0 = [0.0], T = 1.0, quadratures = Int[], lower = zeros(2), upper = ones(2), initial = zeros(2)
    )
    @test_throws ErrorException CorleoneGame.validate(bad, Options())
    nonzero = Game(;
        name = "Bad quadrature", state_labels = ["x"], control_labels = ["A"], owners = [1],
        dynamics! = (dx, x, u, t) -> fill!(dx, 0), objective = (x, i) -> x[2],
        x0 = [0.0, 1.0], T = 1.0, quadratures = [2], lower = [0.0], upper = [1.0], initial = [0.0]
    )
    @test_throws ErrorException CorleoneGame.validate(nonzero, Options())
    @test_throws ErrorException simulate(g, fill(2.0, 3, 4)) # controls outside bounds
    @test_throws ErrorException simulate(g, fill(0.5, 2, 4)) # wrong control dimension
    c = [0.1 0.8 0.4; 0.7 0.2 0.6]
    @test CorleoneGame.transfer_controls(c, 6) ≈ repeat(c; inner = (1, 2))
end

@testset "Independent simulation and layer transcription agree" begin
    g = analytic_game(; T = 2.0)
    controls = [0.0 0.8 0.2 0.0; 0.7 0.0 0.1 0.9; 0.4 0.4 0.5 0.3]
    dense = simulate(g, controls)
    h = g.T / 4
    @test dense.x[1, end] ≈ h * sum(controls[1, :] .+ 2controls[2, :] .- controls[3, :]) atol = 1.0e-9
    @test dense.objectives[2] ≈ h * sum((controls[3, :] .- 0.4) .^ 2) atol = 1.0e-9
    @test first(dense.t) == 0 && last(dense.t) ≈ g.T
    layer, p, st = setup_layer(g, controls, 1)
    shot, _ = layer(nothing, p, st)
    @test isempty(Corleone.shooting_constraints(shot))
    @test maximum(abs.(last(shot.u)[1:3] - dense.x[:, end])) < 1.0e-7
    @test constraint_norms(g, dense, controls) == [0.0, 0.0]
end

@testset "Best responses with multiple control components" begin
    g = analytic_game()
    options = Options(intervals = 24, refined_intervals = 24)
    c = initial_controls(g, 24)
    for player in 1:2
        response = best_response(g, c, player; options)
        owned = findall(==(player), g.owners)
        @test response.status == "Success"
        @test maximum(abs.(response.controls[owned, :] .- [0.2, 0.3, 0.4][owned])) < 1.0e-4
        @test response.controls[findall(!=(player), g.owners), :] == c[findall(!=(player), g.owners), :]
        c = response.controls
    end
end

@testset "Equilibrium, refinement, and nonconvergence reporting" begin
    g = analytic_game()
    o = Options(intervals = 2, refined_intervals = 4, max_rounds = 3, damping = 1.0, extra_starts = Float64[])
    result = solve_game(g; options = o)
    @test result.converged
    @test length(result.runs) == 2
    @test size(last(result.runs).controls) == (3, 4)
    @test last(result.runs).controls ≈ repeat([0.2, 0.3, 0.4], 1, 4) atol = 1.0e-4
    @test maximum(last(result.runs).check.normalized) <= o.tolerance
    run = equilibrium(g; options = Options(intervals = 2, refined_intervals = 2, max_rounds = 1, tolerance = 1.0e-12))
    @test !run.converged
    @test run.controls == last(run.snapshots).controls
    @test run.check.trajectory.objectives == run.trajectory.objectives
    # CSV signs/dimensions and plot generation without a full equilibrium.
    mktempdir() do dir
        CorleoneGame.write_outputs(g, [run], nothing, dir, Options())
        @test isfile(joinpath(dir, "timeseries.png")) && isfile(joinpath(dir, "convergence.png"))
        headers = split(first(readlines(joinpath(dir, "trajectory.csv"))), ',')
        @test length(headers) == 5 + length(g.x0)
        @test length(unique(headers)) == length(headers)
        @test occursin("converged=false", read(joinpath(dir, "report.txt"), String))
    end
end

@testset "run_game writes per-round records" begin
    g = analytic_game()
    o = Options(intervals = 2, refined_intervals = 2, max_rounds = 2, damping = 1.0, extra_starts = Float64[])
    mktempdir() do dir
        result = run_game(g; options = o, folder = dir)
        @test result.converged
        rounds = length(only(result.runs).history)
        @test length(readlines(joinpath(dir, "iterations.csv"))) == 1 + rounds
        @test isfile(joinpath(dir, "iterations", "controls_n2_iter001.csv"))
        @test all(isfile(joinpath(dir, f)) for f in ("trajectory.csv", "controls.csv", "report.txt"))
    end
end

@testset "Fixed-time point costs" begin
    # Reach x(0.5) = 0.25 with a small effort penalty; the cost after t = 0.5 is effort only.
    g = Game(;
        name = "Via point", state_labels = ["x"], control_labels = ["u"], owners = [1],
        dynamics! = (dx, x, u, t) -> (dx[1] = u[1]; dx[2] = 1.0e-3 * u[1]^2),
        objective = (x, i) -> x[2],
        point_costs = PointCost[PointCost(; player = 1, time = 0.5, cost = x -> (x[1] - 0.25)^2)],
        x0 = [0.0, 0.0], T = 1.0, quadratures = [2], lower = [0.0], upper = [1.0], initial = [0.7]
    )
    tr = simulate(g, fill(0.7, 1, 4))
    @test tr.objectives[1] ≈ (0.35 - 0.25)^2 + 1.0e-3 * 0.49 atol = 1.0e-9
    @test_throws ErrorException simulate(g, fill(0.7, 1, 3)) # t = 0.5 is not a grid node
    response = best_response(g, fill(0.7, 1, 4), 1; options = Options(intervals = 4, refined_intervals = 4))
    @test response.status == "Success"
    k = findfirst(t -> isapprox(t, 0.5), response.trajectory.t)
    @test abs(response.trajectory.x[1, k] - 0.25) < 1.0e-3
    @test maximum(response.controls[1, 3:4]) < 1.0e-3 # interior-point solution near the bound
end

@testset "Periodic-return constraints and forcing clocks" begin
    cs = periodic_constraints([1.0, 2.0], 2; tolerance = 0.1)
    @test length(cs) == 8
    @test all(c -> !c.path && startswith(c.label, "!"), cs)
    lower = filter(c -> c.player == 2 && c.index == 2 && isfinite(c.lower), cs)
    @test only(lower).lower ≈ 1.9
    logcs = periodic_constraints([1.0], 1; tolerance = 0.5, log_indices = [1])
    @test only(filter(c -> isfinite(c.upper), logcs)).upper ≈ log(1.5)
    @test_throws ErrorException periodic_constraints([1.0], 2; tolerance = -1.0)
    @test_throws ErrorException periodic_constraints([1.0], 2; tolerance = NaN)
    @test_throws ErrorException periodic_constraints([1.0], 2; tolerance = 1.0, log_indices = [1])
    @test_throws ErrorException validate_periodic_clock(1.0e-14, 1.0, 0.2)
    @test_throws ErrorException validate_periodic_clock(1.5, 1.0, 0.2)
    @test_throws ErrorException validate_periodic_clock(1.0, 1.0, 1.0)
    @test validate_periodic_clock(2.0, 1.0, 0.2) === nothing
end
