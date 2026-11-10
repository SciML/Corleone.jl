using Corleone
using LuxCore
using Random
using Test

struct CountingStage <: LuxCore.AbstractLuxLayer end

LuxCore.initialparameters(::AbstractRNG, ::CountingStage) = (increment = 2,)
LuxCore.initialstates(::AbstractRNG, ::CountingStage) = (count = 0,)

function (::CountingStage)(input, ps, st)
    return input + ps.increment, (count = st.count + 1,)
end

@testset "Stages preserves updated layer states" begin
    rng = MersenneTwister(7)
    for stages in (
            Stages(),
            Stages(CountingStage()),
            Stages(CountingStage(), StageTransition((input, p) -> 3input), CountingStage()),
        )
        ps, st = LuxCore.setup(rng, stages)
        output, next_st = stages(1, ps, st)
        expected = isempty(stages.layers) ? 1 : length(stages.layers) == 1 ? 3 : 11
        @test output == expected
        @test keys(next_st) == keys(st)
        for (layer, old, updated) in zip(stages.layers, values(st), values(next_st))
            if layer isa CountingStage
                @test old.count == 0
                @test updated.count == 1
            else
                @test updated == old
            end
        end

        output_again, final_st = stages(1, ps, next_st)
        @test output_again == expected
        for (layer, updated) in zip(stages.layers, values(final_st))
            layer isa CountingStage && @test updated.count == 2
        end
    end
end

@testset "Stage connector bounds dimensions" begin
    identity_map = (x, p) -> x
    for bounds in ((Float64[], [1.0]), ([0.0], Float64[]))
        @test_throws DimensionMismatch StageTransition(identity_map; p = [0.5], bounds)
        @test_throws DimensionMismatch StageLinkage([0.5]; bounds_u0 = bounds)
        @test_throws DimensionMismatch StageLinkage([0.5]; θ = [0.5], bounds_θ = bounds)
        @test_throws DimensionMismatch StageLinkage([0.5]; bounds)
    end
    @test StageTransition(identity_map).bounds == (Float64[], Float64[])
    @test StageTransition(identity_map; p = [0.5]).bounds == ([-Inf], [Inf])
    @test StageLinkage([0.5]).bounds == ([0.0], [0.0])
end
