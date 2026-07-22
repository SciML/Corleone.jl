# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------

@testset "Controls" begin
    prob = LotkaVolterra.generate()
    sys = symbolic_container(prob.f)

    # 5-breakpoint grid spanning tspan  DimensionMismatch: new dimensions (6,) must be consistent with array length 2
    cgrid = collect(LinRange(0.0, 12.0, 6))
    pc1 = PiecewiseParameter(:u1, copy(cgrid))
    pc2 = PiecewiseParameter(:u2, copy(cgrid))

    @testset "Construction" begin
        c = Controls(pc1, pc2; sys = sys)
        reset!(c)
        @test length(c.controls) == 2
        @test length(c.permutation) == 2
    end

    @testset "collect_timegrid covers tspan" begin
        c = Controls(pc1, pc2; sys = sys)
        reset!(c)
        _, st = LuxCore.setup(rng, c)
        tspans = Corleone.collect_timegrid(c, nothing, st, (0.0, 12.0))
        @test first(first(tspans)) == 0.0
        @test last(last(tspans)) == 12.0
    end

    @testset "Evaluation returns vector matching ncontrols" begin
        c = Controls(pc1, pc2; sys = sys)
        reset!(c)
        ps, st = LuxCore.setup(rng, c)
        result, _ = c(0.5, ps, st)
        @test length(result) == 2
    end
end

# ---------------------------------------------------------------------------
# Controls – maybekeys gensym branch
# ---------------------------------------------------------------------------

@testset "Controls – maybekeys gensym branch" begin
    prob = LotkaVolterra.generate()
    sys = symbolic_container(prob.f)
    # Integer id (no symbolic name) → maybekeys hits gensym branch
    pc1 = PiecewiseParameter(5, collect(LinRange(0.0, 12.0, 4)))  # id=5 (param index)
    pc2 = PiecewiseParameter(6, collect(LinRange(0.0, 12.0, 4)))  # id=6
    c = Controls(pc1, pc2; sys = sys)
    @test length(c.controls) == 2
end
