# Reproducible synthetic observations from a saved trajectory.csv.
using CorleoneGame, Test

@testset "Synthetic observations" begin
    g = Game(;
        name = "Synthetic", state_labels = ["x", "y"], control_labels = ["u"], owners = [1],
        dynamics! = (dx, x, u, t) -> fill!(dx, 0), objective = (x, i) -> x[1],
        x0 = [0.0, 0.0], T = 1.0, quadratures = Int[], lower = [0.0], upper = [1.0], initial = [0.0]
    )
    mktempdir() do dir
        # x = t and y = 2 - t on a coarse grid; linear interpolation is exact.
        open(joinpath(dir, "trajectory.csv"), "w") do io
            println(io, "time,x_1,x_2,u_A")
            for t in 0:0.25:1
                println(io, join((t, t, 2 - t, 0.0), ','))
            end
        end
        times = [0.0, 0.3, 0.8, 1.0]
        exact = create_synthetic_data(g, dir; sigma = 0.0, measurement_times = times, output = "exact.csv")
        headers, data = CorleoneGame.read_output_csv(exact)
        @test headers == ["time", "x_1", "x_2", "sigma", "seed"]
        @test data[1, :] == times
        @test data[2, :] ≈ times atol = 1.0e-12
        @test data[3, :] ≈ 2 .- times atol = 1.0e-12
        a = create_synthetic_data(g, dir; sigma = [0.1, 0.2], seed = 7, output = "a.csv")
        b = create_synthetic_data(g, dir; sigma = [0.1, 0.2], seed = 7, output = "b.csv")
        c = create_synthetic_data(g, dir; sigma = [0.1, 0.2], seed = 8, output = "c.csv")
        @test read(a) == read(b)
        @test read(a) != read(c)
        @test size(CorleoneGame.read_output_csv(a)[2], 2) == 20
        @test_throws ErrorException create_synthetic_data(g, dir; sigma = [0.1, 0.2, 0.3])
    end
end
