using Test, Corleone, LuxCore, Random, OrdinaryDiffEqTsit5, ComponentArrays
using Distributions: Normal, LogNormal, logpdf
using TransformVariables: asℝ₊
using ForwardDiff, ReverseDiff, Zygote

@testset "Experiments merges values and ShootingLayer owns execution" begin
    calls = Ref(0)
    initialize = (rng, n) -> (calls[] += 1; fill(0.7, n))
    prior = Normal(0.6, 0.2)
    problem = ODEProblem((u, p, t) -> [-p[1] * u[1] + p[2]], [1.0], (0.0, 1.0), [0.7, 0.2])
    base = ShootingLayer(
        problem, Tsit5(), Tunable(1; default = initialize, prior) => nothing,
        Tunable(1; kind = Val(:variable), lowerbound = 0.0) => [0.0]; controls = [2]
    )
    batch = Experiments(
        base, (
            a = (;), b = (
                annotations = (
                    Constant(2; default = [0.3, 0.4]) => [0.5],
                    Tunable(1; kind = Val(:variable), default = 2.0, transform = asℝ₊, prior = LogNormal(0.0, 1.0)) => [0.0],
                ),
            ),
        )
    )
    @test batch.problem === problem
    @test !hasproperty(batch, :shooting)
    @test Experiments(problem; controls = [2], experiments = (a = (;),)).problem === problem
    @test !(:ExperimentLayer in names(Corleone))
    @test !hasproperty(batch.experiments.b, :shooting)
    @test !hasproperty(batch.experiments.b, :base)
    ps, st = LuxCore.setup(Xoshiro(1), batch)
    @test calls[] == 1
    @test length(ComponentArray(ps.shared)) == 3
    @test isempty(ComponentArray(ps.experiments.a))
    @test length(ComponentArray(ps.experiments.b)) == 1
    @test LuxCore.parameterlength(batch) == length(ComponentArray(ps)) == 4
    @test batch.experiments.b(nothing, ps.experiments.b, st.experiments.b) === (ps.experiments.b, st.experiments.b)
    @test st.execution.a.active !== st.execution.b.active
    @test 0.5 in st.execution.b.timestops
    @test !(0.5 in st.execution.a.timestops)
    merged_calls, same_st = batch(nothing, ps, st)
    @test same_st === st
    @test merged_calls isa Tuple
    @test length(merged_calls) == 2
    @test length(merged_calls[1]) == 2
    y, next = base(batch, ps, st)
    @test y isa Vector
    @test length(y) == 2
    @test first(base(nothing, merged_calls[1]...)).u ≈ y[1].u
    @test first(base(nothing, merged_calls[2]...)).u ≈ y[2].u
    @test first(ShootingLayer(problem, Tsit5())(batch, ps, st))[1].u ≈ y[1].u
    yp, _ = base(batch, ComponentArray(ps), st)
    @test y[1].u ≈ yp[1].u
    @test y[2].u ≈ yp[2].u
    @test first(y[1].u) ≈ [1.0]
    @test first(y[2].u) ≈ [2.0]
    @test keys(next.experiments) == (:a, :b)
    # The global control is used by a, while b has a fixed local replacement.
    ps.shared.parameter.parameters.p₂.parameter.ps .= 0.9
    changed, _ = base(batch, ps, st)
    @test last(changed[1].u) != last(y[1].u)
    @test changed[2].u ≈ y[2].u
    ps.experiments.b.initialcondition.initialconditions.u₁.parameter.ps .= log(3.0)
    local_changed, _ = base(batch, ps, st)
    @test local_changed[1].u ≈ changed[1].u
    @test first(local_changed[2].u) ≈ [3.0]
    ps.shared.parameter.parameters.p₁.parameter.ps .= 0.8
    shared_changed, _ = base(batch, ps, st)
    @test last(shared_changed[1].u) != last(local_changed[1].u)
    @test last(shared_changed[2].u) != last(local_changed[2].u)
    @test calls[] == 1
    expected = logpdf(prior, 0.8) + logpdf(LogNormal(0.0, 1.0), 3.0) + log(3.0)
    @test log_prior(batch, ps) ≈ expected
    @test log_prior(batch, ComponentArray(ps)) ≈ expected
    lb, ub = get_bounds(batch, ComponentArray(ps), st)
    @test collect(ComponentArray(lb)) == [0.0, -Inf, -Inf, -Inf]
    @test collect(ComponentArray(ub)) == fill(Inf, 4)
    ordinary = Experiments(base, (a = (;), b = (annotations = (Tunable(1; default = 0.9) => nothing,),)))
    ordinary_ps, ordinary_st = LuxCore.setup(Xoshiro(1), ordinary)
    ordinary_y, _ = base(ordinary, ordinary_ps, ordinary_st)
    @test last(ordinary_y[1].u) != last(ordinary_y[2].u)
    @test length(ComponentArray(ordinary_ps.experiments.b)) == 1
    @test_throws ArgumentError Experiments(base, (a = (annotations = (Tunable(2) => [0.7, 0.3],),),))
    @test_throws ArgumentError Experiments(base, (a = (annotations = (Tunable(2) => [1.5],),),))
    @test_throws ArgumentError Experiments(base, (a = (annotations = (Tunable(2) => nothing, Constant(2) => nothing),),))
    @test_throws DimensionMismatch Experiments(base, (a = Experiment(1 => ([0.5], [1.0]); u0 = [1.0, 2.0]),))
    other = ShootingLayer(ODEProblem(problem.f, problem.u0, (0.0, 2.0), problem.p), Tsit5())
    @test_throws ArgumentError other(batch, ps, st)
    @test_throws ArgumentError Experiments(base, (;))
    @test_throws ArgumentError Experiments(base, (a = (unexpected = 1,),))
end

@testset "Unused global unknowns are omitted and local groups are retained" begin
    n = 3
    problem = ODEProblem((u, p, t) -> p, zeros(n), (0.0, 1.0), ones(n))
    definitions = (
        Tunable(1:n; kind = Val(:variable), default = ones(n)) => [0.0, 0.5],
        Tunable(1:n; default = ones(n)) => nothing,
    )
    base = ShootingLayer(problem, Tsit5(), definitions...; controls = collect(1:n))
    spec = (annotations = definitions,)
    batch = Experiments(base, (a = spec, b = spec))
    ps, st = LuxCore.setup(Xoshiro(2), batch)
    @test isempty(ComponentArray(ps.shared))
    @test length(batch.experiments.a.parameter.parameters) == 1
    @test length(batch.experiments.a.initialcondition.initialconditions) == 1
    @test get_number_of_shooting_constraints(batch, ps, st) == 6
    @test length(ComponentArray(ps)) == 18
    @test ps.experiments.a.parameter.parameters.p₁.parameter.p₁.ps !== ps.experiments.b.parameter.parameters.p₁.parameter.p₁.ps
    y, next = base(batch, ComponentArray(ps), st)
    @test all(next.experiments.a.initialcondition.active)
    @test last(y[1].u) ≈ fill(1.5, n)
    @test y[1].u ≈ y[2].u
    @test length(shooting_constraints(y[1])) == 3
    @test length(ComponentArray(first(get_bounds(batch, ps, st)))) == length(ComponentArray(ps))
end

@testset "AD through set-level parameter merging" begin
    problem = ODEProblem((u, p, t) -> [p[1] + p[2]], [1.0], (0.0, 1.0), [0.7, 0.2])
    base = ShootingLayer(problem, Tsit5(), Tunable(1; kind = Val(:variable)) => [0.0]; controls = [2])
    locals = (Tunable(1; kind = Val(:variable)) => [0.0], Tunable(2) => nothing)
    batch = Experiments(base, (a = (annotations = locals,), b = (annotations = locals,)))
    ps, st = LuxCore.setup(Xoshiro(1), batch)
    q, reconstruct = pack(default_vectorizer(), ps)
    function loss(q)
        parameters = reconstruct(q)
        sum(keys(batch.experiments)) do name
            config = batch.configurations[name]
            local_ps, local_st = batch.experiments[name](nothing, getproperty(parameters.experiments, name), st.experiments[name])
            merged = Corleone.merge_experiment_ps(parameters.shared, local_ps, st.shared, local_st)
            state = Corleone.merge_experiment_st(config, st.shared, local_st, st.execution[name].active)
            p, _ = config.parameter(0.0, merged.parameter, state.parameter)
            (u, _), _ = config.initialcondition(([0.0], 0.0), merged.initialcondition, state.initialcondition)
            sum(abs2, p) + sum(abs2, u)
        end
    end
    expected = [4q[1]; 2 .* q[2:end]]
    @test ForwardDiff.gradient(loss, q) ≈ expected
    @test ReverseDiff.gradient(loss, q) ≈ expected
    @test only(Zygote.gradient(loss, q)) ≈ expected
end

@testset "Global fixed values and global initial variables" begin
    calls = Ref(0)
    initialize = (rng, n) -> (calls[] += 1; fill(0.7, n))
    problem = ODEProblem((u, p, t) -> [p[1] + p[2]], [1.0], (0.0, 1.0), [0.7, 0.2])
    base = ShootingLayer(problem, Tsit5(), Constant(1; default = initialize) => nothing; controls = [2])
    batch = Experiments(base, (a = (u0 = [2.0],), b = (u0 = [3.0],)))
    ps, st = LuxCore.setup(Xoshiro(1), batch)
    @test calls[] == 1
    @test length(ComponentArray(ps.shared)) == 1
    @test isempty(ComponentArray(ps.experiments))
    y, _ = base(batch, ComponentArray(ps), st)
    @test first(y[1].u) ≈ [2.0]
    @test first(y[2].u) ≈ [3.0]
    @test last(y[1].u) ≈ [2.9]
    @test last(y[2].u) ≈ [3.9]
    @test calls[] == 1
    # Updating a global fixed value in st is visible in both experiments.
    st.shared.parameter.parameters.p₁.parameter.p₁.parameters.ps .= 1.0
    changed, _ = base(batch, ps, st)
    @test last(changed[1].u) ≈ [3.2]
    @test last(changed[2].u) ≈ [4.2]
end

@testset "Unused global priors and initializers are not evaluated" begin
    fail_if_used = (rng, n) -> error("Unused global initializer executed")
    problem = ODEProblem((u, p, t) -> [p[1]], [1.0], (0.0, 1.0), [1.0])
    base = ShootingLayer(
        problem, Tsit5(),
        Tunable(1; kind = Val(:variable), default = fail_if_used, prior = Normal(100.0, 1.0)) => [0.0],
        Tunable(1; default = fail_if_used, prior = Normal(100.0, 1.0)) => nothing; controls = [1]
    )
    spec = (annotations = (Tunable(1; kind = Val(:variable), default = 1.0) => [0.0], Tunable(1; default = 1.0) => nothing),)
    batch = Experiments(base, (a = spec, b = spec))
    ps, st = LuxCore.setup(Xoshiro(1), batch)
    @test isempty(ComponentArray(ps.shared))
    @test collect(ComponentArray(ps)) == ones(4)
    @test log_prior(batch, ps) == 0.0
    @test last(first(base(batch, ps, st))[1].u) ≈ [2.0]
end

@testset "A local tunable replaces a global constant" begin
    problem = ODEProblem((u, p, t) -> [p[1] + p[2]], [1.0], (0.0, 1.0), [0.7, 0.2])
    base = ShootingLayer(problem, Tsit5(), Constant(2) => nothing; controls = [2])
    batch = Experiments(base, (a = (;), b = (annotations = (Tunable(2; default = 0.4) => nothing,),)))
    ps, st = LuxCore.setup(Xoshiro(1), batch)
    y, _ = base(batch, ComponentArray(ps), st)
    @test last(y[1].u) ≈ [1.9]
    @test last(y[2].u) ≈ [2.1]
    ps.experiments.b.parameter.parameters.p₂.parameter.ps .= 0.5
    changed, _ = base(batch, ps, st)
    @test changed[1].u ≈ y[1].u
    @test last(changed[2].u) ≈ [2.2]
end

@testset "Trajectory bounds retain shared transforms and local replacements" begin
    problem = ODEProblem((u, p, t) -> [-p[1] * u[1]], [1.0], (0.0, 1.0), [0.7])
    experiments = Experiments(
        problem, Tunable(1; transform = asℝ₊) => nothing;
        bounds = Dict(:p₁ => (0.0, 1.0)), experiments = (
            a = (;),
            b = (annotations = (Constant(1; default = 1.2, lowerbound = 0.1, upperbound = 2.0) => nothing,),),
        )
    )
    ps, st = LuxCore.setup(Xoshiro(1), experiments)
    trajectories, _ = ShootingLayer(problem, Tsit5())(experiments, ps, st)
    @test get_bounds(trajectories[1].sys, :p₁) == (-Inf, Inf)
    @test get_bounds(trajectories[2].sys, :p₁) == (0.1, 2.0)
    @test last(trajectories[1].u) ≈ [exp(-0.7)] rtol = 1.0e-4
    @test last(trajectories[2].u) ≈ [exp(-1.2)] rtol = 1.0e-4
end
