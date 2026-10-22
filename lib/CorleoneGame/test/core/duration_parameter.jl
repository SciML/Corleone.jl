# Player-owned scalar duration (free-time games on normalized time).
using CorleoneGame, Test
function duration_test_dynamics!(dx, x, p, s)
    dx[1] = p[3] * (p[1] + p[2])
    return dx[2] = p[3] * p[2]^2
end
g = Game(
    name = "Duration test", state_labels = ["position"], control_labels = ["A", "B"], owners = [1, 2],
    dynamics! = duration_test_dynamics!, objective = (x, i, p) -> i == 1 ? p[1] : x[2],
    constraints = [StateConstraint(player = i, index = 1, label = "arrival", lower = 1.0, upper = 1.0) for i in 1:2],
    x0 = [0.0, 0.0], T = 1.0, quadratures = [2], lower = zeros(2), upper = ones(2), initial = [0.5, 0.5],
    parameters = [1.0], parameter_owners = [1], parameter_labels = ["duration"], bounds_p = ([0.2], [4.0]), time_parameter = 1
)
@testset "Scalar duration ownership and matching" begin
    for shooting in (:single, :multiple)
        o = Options(; shooting, intervals = 4, refined_intervals = 4, shooting_intervals = 2, nlp_solver = :ipopt)
        u = initial_controls(g, 4)
        a = best_response(g, u, 1; options = o)
        @test a.status == "Success"
        @test a.parameters[1] ≈ 2 / 3 atol = 2.0e-6
        @test size(a.controls) == (2, 4)
        b = best_response(g, a.controls, 2; options = o, parameters = a.parameters)
        @test b.parameters == a.parameters
        @test b.matching < 1.0e-6
        @test last(CorleoneGame.physical_times(g, b.trajectory)) ≈ 2 / 3 atol = 2.0e-6
    end
end
@testset "Duration persists through rounds, physical CSVs, and replotting" begin
    o = Options(intervals = 4, refined_intervals = 4, max_rounds = 2, damping = 1.0, extra_starts = Float64[])
    result = solve_game(g; options = o)
    run = last(result.runs)
    @test run.parameters[1] ≈ 2 / 3 atol = 2.0e-6
    @test run.trajectory.parameters == run.parameters
    @test all(s.parameters == s.trajectory.parameters for s in run.snapshots)
    mktempdir() do dir
        CorleoneGame.write_outputs(g, result.runs, nothing, dir, o)
        headers, data = CorleoneGame.read_output_csv(joinpath(dir, "trajectory.csv"))
        @test data[findfirst(==("time"), headers), end] ≈ run.parameters[1]
        @test data[findfirst(==("normalized_time"), headers), end] == 1.0
        @test all(data[findfirst(==("parameter_1"), headers), :] .== run.parameters[1])
        saved = CorleoneGame.read_plot_outputs(g, dir)
        @test saved.trajectory.parameters == run.parameters
        @test saved.controls == run.controls
        replot_outputs(g; folder = dir)
        @test filesize(joinpath(dir, "timeseries.png")) > 1000
        @test length(CorleoneGame.constraint_plot_series(g, run.trajectory)) == 1
    end
end
