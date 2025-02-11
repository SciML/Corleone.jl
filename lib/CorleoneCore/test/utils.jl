using CorleoneCore
using Test

@testset "Sorting" begin 
    timepoints = rand(1000)
    timepoints_sorted = sort(unique(timepoints))
    @test CorleoneCore.maybe_unique_sort(timepoints) == timepoints_sorted
    @test begin
        CorleoneCore.maybe_unique_sort!(timepoints)
        timepoints == timepoints_sorted
    end
end

@testset "Copy" begin 
    timepoints = rand(1000)
    @test CorleoneCore._maybecopy(timepoints) == timepoints
    @test !(CorleoneCore._maybecopy(timepoints) === timepoints)
    timepoints = (; a = randn(10), b = randn(5))
    @test CorleoneCore._maybecopy(timepoints) == timepoints
    @test !(CorleoneCore._maybecopy(timepoints) === timepoints)
end

@testset "Timepoint collections" begin 
    timepoints = LinRange(0.0, 10., 25)
    tspan = (0.0, 10.0)

    config = (; 
        layer_2 = (; 
            layer_3 = (; 
                tstops = timepoints[5:10]
            ), 
            saveat = timepoints[11:20],
        ),
        layer_1 = (; 
            tstops = timepoints[1:5], 
            layer_4 = (; 
                saveat = timepoints[21:25],
                tstops = timepoints[21:25]
            )
        )
    )

    @test collect_tstops(config, tspan) ≈ vcat(timepoints[1:10], timepoints[21:25])
    @test collect_saveat(config, tspan) ≈ vcat(timepoints[11:20], timepoints[21:25])
end
