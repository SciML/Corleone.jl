# Replotting saved CSV output without optimization.
using CorleoneGame, Test

function plotting_game(; label = "Population", replay = true)
    rhs! = replay ? ((dx, x, u, t) -> (dx[1] = u[1]; dx[2] = u[1]^2)) :
        ((dx, x, u, t) -> error("Direct CSV plotting must not evaluate dynamics"))
    return Game(;
        name = "CSV plotting", state_labels = [label], control_labels = ["Effort"], owners = [1],
        dynamics! = rhs!, objective = (x, i) -> x[2], x0 = [1.0, 0.0], T = 1.0, quadratures = [2],
        lower = [0.0], upper = [1.0], initial = [0.2], constraints = [
            StateConstraint(; player = 1, index = 1, label = "floor", lower = 0.5, path = true),
            StateConstraint(; player = 1, index = 2, label = "budget", upper = 1.0),
        ]
    )
end

@testset "Replot saved CSVs without optimizing" begin
    g = plotting_game()
    controls = reshape([0.1, 0.3, 0.2, 0.4], 1, :)
    trajectory = simulate(g, controls)
    history = [(; round = k, objectives = trajectory.objectives, gains = [0.1 / k], norms = [0.0]) for k in 1:2]
    run = (;
        trajectory, controls, history, check = (; normalized = [0.0], norms = [0.0]),
        snapshots = [(; trajectory, controls)], converged = true,
    )
    mktempdir() do dir
        CorleoneGame.write_outputs(g, [run], nothing, dir, Options())
        unchanged = ["trajectory.csv", "iterations.csv", "controls.csv", "report.txt", "iterations/timeseries_n4_iter001.png"]
        original = Dict(file => read(joinpath(dir, file)) for file in unchanged)
        renamed = plotting_game(; label = "Renamed population", replay = false)
        saved = CorleoneGame.read_plot_outputs(renamed, dir)
        @test saved.controls == controls
        @test saved.trajectory.x == trajectory.x
        @test saved.trajectory.physical == trajectory.physical[1:1, :]
        @test length(saved.runs) == 1
        @test last(saved.runs[1].history).gains == [0.05]
        files = replot_outputs(renamed; folder = dir)
        @test all(path -> filesize(path) > 1000, values(files))
        @test all(file -> read(joinpath(dir, file)) == original[file], unchanged)
        # Remove new columns to exercise legacy forward replay and its check.
        path = joinpath(dir, "trajectory.csv")
        lines = readlines(path)
        keep = findall(h -> !startswith(h, "numerical_x_"), split(first(lines), ','))
        write(path, join([join(split(line, ',')[keep], ',') for line in lines], "\n") * "\n")
        legacy = CorleoneGame.read_plot_outputs(g, dir)
        @test legacy.controls == controls
        @test legacy.trajectory.x ≈ trajectory.x
        # Changed parameters/dynamics must not silently overwrite plots from old runs.
        lines = readlines(path)
        cells = split(lines[2], ','); cells[2] = "9.0"; lines[2] = join(cells, ',')
        write(path, join(lines, "\n") * "\n")
        @test_throws ErrorException CorleoneGame.read_plot_outputs(g, dir)
        rm(path)
        @test_throws ErrorException replot_outputs(g; folder = dir)
    end
end
