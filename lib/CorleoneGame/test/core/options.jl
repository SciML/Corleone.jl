# Grids, solver choices, primal warm starts, fallback accounting, and audits.
using CorleoneGame, Test, Corleone

function analytic_game(; target = [0.2, 0.3, 0.4], bound = 1.0, cap = Inf, owners = [1, 1, 2])
    function rhs!(dx, x, u, t)
        dx[1] = u[1] + 2u[2] - u[3]
        dx[2] = (u[1] - target[1])^2 + (u[2] - target[2])^2
        return dx[3] = (u[3] - target[3])^2
    end
    return Game(;
        name = "Analytic options",
        state_labels = ["x"],
        control_labels = ["A1", "A2", "B1"], owners,
        dynamics! = rhs!,
        objective = (x, i) -> x[i + 1],
        constraints = isfinite(cap) ? [StateConstraint(player = 1, index = 1, label = "cap", upper = cap)] : StateConstraint[],
        x0 = zeros(3), T = 1.3, quadratures = [2, 3],
        lower = zeros(3), upper = fill(bound, 3), initial = fill(min(0.7, bound), 3)
    )
end

@testset "Equidistant grids and control reconstruction" begin
    g = analytic_game()
    for (ns, n) in ((1, 6), (2, 6), (3, 6), (4, 8))
        o = Options(shooting = :multiple, shooting_intervals = ns, intervals = n, refined_intervals = n)
        c = reshape(collect(range(0.05, 0.95; length = 3n)), 3, n)
        layer, p, st = setup_layer(g, c, 1; options = o)
        @test length(propertynames(p)) == ns
        @test CorleoneGame.unpack_controls(g, p, :multiple, n) ≈ c
        shot, _ = layer(nothing, p, st)
        @test maximum(abs, Corleone.shooting_constraints(shot); init = 0.0) < 1.0e-8
        @test last(shot.u)[1:3] ≈ simulate(g, c).x[:, end] atol = 1.0e-8
    end
    @test_throws ErrorException setup_layer(
        g, initial_controls(g, 5), 1;
        options = Options(shooting = :multiple, shooting_intervals = 2, intervals = 6, refined_intervals = 6)
    )
    @test_throws ErrorException CorleoneGame.validate(g, Options(extra_starts = [1.1]))
    @test_throws ErrorException CorleoneGame.validate(g, Options(nlp_solver = :unknown))
    @test_throws ErrorException CorleoneGame.validate(g, Options(nlp_solver = :blocksqp))
    c = [0.1 0.8 0.4; 0.7 0.2 0.6]
    @test CorleoneGame.transfer_controls(c, 6) ≈ repeat(c; inner = (1, 2))
    @test vec(sum(CorleoneGame.transfer_controls(c, 5); dims = 2)) / 5 ≈ vec(sum(c; dims = 2)) / 3
end

@testset "Solver choices and primal warm starts" begin
    g = analytic_game()
    for (shooting, solver) in ((:single, :ipopt), (:single, :uno), (:multiple, :ipopt), (:multiple, :uno), (:multiple, :blocksqp))
        o = Options(;
            shooting, nlp_solver = solver, fallback_solver = nothing, shooting_intervals = 2,
            intervals = 4, refined_intervals = 4, max_iters = 100, extra_starts = Float64[], warm_start = true
        )
        workspace = Workspace()
        c = initial_controls(g, 4)
        a = best_response(g, c, 1; options = o, workspace)
        @test !a.warm_started
        @test a.solver == solver
        @test a.controls[1:2, :] ≈ repeat([0.2, 0.3], 1, 4) atol = 1.0e-4
        # Change objective coefficients, opponent policy, bounds, and constraints.
        updated = analytic_game(target = [0.1, 0.25, 0.4], bound = 0.6, cap = 0.065)
        c = fill(0.5, 3, 4)
        b = best_response(updated, c, 1; options = o, workspace)
        @test b.warm_started
        @test b.controls[1:2, :] ≈ repeat([0.09, 0.23], 1, 4) atol = 1.0e-4
        @test b.controls[3, :] == c[3, :]
        @test b.matching < o.feasibility_tolerance
        @test constraint_norms(updated, b.trajectory, b.controls)[1] <= o.feasibility_tolerance
        extra = best_response(updated, c, 1; options = o, workspace, start = 0.8)
        @test !extra.warm_started
        changed_grid = best_response(updated, fill(0.5, 3, 6), 1; options = o, workspace)
        @test !changed_grid.warm_started
        empty!(workspace)
        @test isempty(workspace.responses)
    end
end

@testset "Fallback accounting, audits, and numerical-only solve" begin
    g = analytic_game()
    o = Options(
        intervals = 2, refined_intervals = 2, nlp_solver = :ipopt,
        solver_options = (; max_iter = 0), fallback_solver = :uno, fallback_options = (; max_iterations = 100)
    )
    response = best_response(g, initial_controls(g, 2), 1; options = o)
    @test response.solver == :uno
    @test length(response.attempts) == 2
    @test response.iterations == sum(a.iterations for a in response.attempts)
    workspace = Workspace()
    o = Options(
        shooting = :multiple, nlp_solver = :uno, fallback_solver = nothing,
        shooting_intervals = 2, intervals = 4, refined_intervals = 6, max_rounds = 2,
        damping = 1.0, extra_starts = Float64[], audit_every_round = true, warm_start = false
    )
    result = solve_game(g; options = o, workspace)
    @test result.converged
    @test length(result.runs) == 2
    @test size(last(result.runs).controls, 2) == 6
    @test result.refined_check === nothing # the audit of the transferred coarse policy is disabled in solve_game
    @test isempty(workspace.responses)
    o = Options(intervals = 2, refined_intervals = 2, extra_starts = [0.25])
    check = audit(g, initial_controls(g, 2); options = o, multistart = true)
    @test check.feasible
    @test all(r -> r.trajectory !== nothing, check.responses)
    one = analytic_game(owners = [1, 1, 1])
    @test player_count(one) == 1
    @test best_response(one, initial_controls(one, 2), 1; options = o).status == "Success"
end
