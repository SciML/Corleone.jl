using Test, Corleone, LuxCore, Random, ComponentArrays, OrdinaryDiffEqTsit5
using TransformVariables: asℝ₊

@testset "State-generated experiment traversal infers concrete results" begin
    problem = ODEProblem((u, p, t) -> [-p[1] * u[1] + p[2]], [1.0], (0.0, 1.0), [0.7, 0.2])
    shooting = ShootingLayer(problem, Tsit5())
    definitions = (Tunable(1; kind = Val(:variable)) => [0.0], Tunable(2) => nothing)
    variants = (
        same = (annotations = definitions,),
        fixed_grid = (
            annotations = (
                Tunable(1; kind = Val(:variable), default = [2.0, 1.0]) => [0.0, 0.5],
                Constant(2; default = 0.3) => nothing,
            ),
        ),
        transformed = (
            annotations = (
                Tunable(1; kind = Val(:variable), default = 2.0, transform = asℝ₊) => [0.0],
                Tunable(1; default = 1.1, transform = asℝ₊) => nothing,
            ),
        ),
    )
    for variant in values(variants)
        experiments = Experiments(problem; experiments = (a = (annotations = definitions,), b = variant))
        native, st = LuxCore.setup(Xoshiro(1), experiments)
        for ps in (native, ComponentArray(native))
            calls, prepared = @inferred experiments(nothing, ps, st)
            @test prepared === st
            a, _ = @inferred shooting(nothing, calls[1]...)
            b, _ = @inferred shooting(nothing, calls[2]...)
            trajectories, next = @inferred shooting(experiments, ps, st)
            @test trajectories[1].u ≈ a.u
            @test trajectories[2].u ≈ b.u
            @test eltype(trajectories) === Union{typeof(a), typeof(b)}
            @test all(isconcretetype, Base.uniontypes(eltype(trajectories)))
            @test keys(next.experiments) == (:a, :b)
            @test keys(next.execution) == (:a, :b)
        end
        # Generation uses state schemas, but outputs retain declaration order.
        reordered = merge(
            st, (;
                experiments = (; b = st.experiments.b, a = st.experiments.a),
                execution = (; b = st.execution.b, a = st.execution.a),
            )
        )
        calls, _ = @inferred experiments(nothing, native, reordered)
        trajectories, next = @inferred shooting(experiments, native, reordered)
        @test keys(next.experiments) == (:a, :b)
        @test first(shooting(nothing, calls[1]...)).u ≈ trajectories[1].u
        @test first(shooting(nothing, calls[2]...)).u ≈ trajectories[2].u
        missing = merge(st, (; experiments = (; a = st.experiments.a)))
        @test_throws ArgumentError experiments(nothing, native, missing)
        @test_throws ArgumentError shooting(experiments, native, missing)
    end
end

@testset "Generated traversal handles one and three experiments" begin
    problem = ODEProblem((u, p, t) -> [-p[1] * u[1] + p[2]], [1.0], (0.0, 1.0), [0.7, 0.2])
    shooting = ShootingLayer(problem, Tsit5())
    specs = (
        shared = (annotations = (Tunable(1; kind = Val(:variable)) => [0.0], Tunable(2) => nothing),),
        fixed = (annotations = (Tunable(1; kind = Val(:variable), default = [2.0, 1.0]) => [0.0, 0.5], Constant(2) => nothing),),
        transformed = (annotations = (Tunable(1; kind = Val(:variable), default = 2.0, transform = asℝ₊) => [0.0], Tunable(1; default = 1.1, transform = asℝ₊) => nothing),),
    )
    for definitions in ((only = specs.shared,), specs)
        experiments = Experiments(problem; experiments = definitions)
        ps, st = LuxCore.setup(Xoshiro(1), experiments)
        for parameters in (ps, ComponentArray(ps))
            calls, _ = @inferred experiments(nothing, parameters, st)
            trajectories, next = @inferred shooting(experiments, parameters, st)
            @test length(trajectories) == length(definitions)
            @test length(Base.uniontypes(eltype(trajectories))) == length(definitions)
            @test all(isconcretetype, Base.uniontypes(eltype(trajectories)))
            @test keys(next.experiments) == keys(definitions)
            for i in eachindex(calls)
                direct, _ = @inferred shooting(nothing, calls[i]...)
                @test trajectories[i].u ≈ direct.u
            end
        end
    end
end
