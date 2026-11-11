using Corleone, LuxCore, ComponentArrays, OrdinaryDiffEqTsit5, Test, Random
using ForwardDiff, ReverseDiff, Zygote, RuntimeGeneratedFunctions
using SciMLBase
using DiffEqBase: SensitivityADPassThrough
using Distributions: Normal, logpdf

@testset "Symbolic measurements share the dynamic optimization interface" begin
    function dynamics!(du, u, p, t)
        du[1] = p[1] + p[2]
        return nothing
    end
    problem = ODEProblem(ODEFunction{true, SciMLBase.FullSpecialize}(dynamics!), [0.0], (0.0, 2.0), [1.0, 0.0]; abstol = 1.0e-11, reltol = 1.0e-11, sensealg = SensitivityADPassThrough())
    shooting = ShootingLayer(problem, Tsit5())
    ta, tb = [0.5, 1.0, 1.5], [0.25, 1.25]
    ya, yb = [1.1, 3.2, 4.8], [-0.125, 2.175]
    experiments = Experiments(
        problem, Tunable(1; default = 1.0) => nothing;
        controls = [2], experiments = (
            a = (
                annotations = (Constant(2; default = [0.0, 1.0]) => [1.0],),
                observations = (:(u₁ + p₂ + t) => (ta, ya, 0.5),),
            ),
            b = (
                annotations = (
                    Tunable(1; default = 2.0) => nothing,
                    Constant(2; default = 0.5) => nothing,
                    Tunable(1; kind = Val(:variable), default = 1.0) => [0.0],
                ),
                observations = (:(u₁ - p₁) => (tb, yb, 0.25),),
            ),
        )
    )
    ps, st = LuxCore.setup(Xoshiro(1), experiments)
    model = CorleoneDynamicOptProblem(shooting, experiments; ps, st)
    @test st.execution.a.timestops == [0.0, 1.0, 2.0]
    @test st.execution.b.timestops == [0.0, 2.0]
    samples = sort(vcat(ta, tb))
    @test Corleone.get_saveat(experiments, st) == samples
    @test model.evaluation_grid === st.saveat
    predictions, next = shooting(experiments, ps, st)
    @test predictions[1].t == predictions[2].t == vcat(0.0, samples, 2.0)
    @test length(predictions[1].residuals) == 2
    @test length(predictions[2].residuals) == 1
    @test next.execution.b.timestops == [0.0, 2.0]
    @test next.saveat === st.saveat
    q, reconstruct = pack(default_vectorizer(), ps)
    @test q == [1.0, 1.0, 2.0]
    expected(q) = vcat(
        (ya .- (q[1] .* ta .+ max.(0, ta .- 1) .+ (ta .>= 1) .+ ta)) ./ 0.5,
        (yb .- (q[2] .+ (q[3] + 0.5) .* tb .- q[3])) ./ 0.25,
    )
    @test (@inferred residual(model)) ≈ expected(q)
    @test n_observations(model) == 5
    objective(q) = optimization_objective(model, reconstruct(q))
    @test (@inferred optimization_objective(model)) ≈ sum(abs2, expected(q)) / 2
    gradient = ForwardDiff.gradient(x -> sum(abs2, expected(x)) / 2, q)
    for differentiate in (ForwardDiff.gradient, ReverseDiff.gradient)
        @test differentiate(objective, q) ≈ gradient
    end
    selector = model.objective.data.a.observations[1].selector
    @test first(Zygote.gradient(u -> selector(u, [1.0, 0.5], 1.0), [2.0])) ≈ [1.0]
    packed = ComponentArray(ps)
    @test (@inferred residual(model, packed)) ≈ expected(q)
    @test (@inferred optimization_objective(model, packed)) ≈ sum(abs2, expected(q)) / 2
    @test get_bounds(model.layer, packed, st) == get_bounds(experiments, packed, st)
    @test model.ps === ps && model.st === st
    reordered = merge(st, (; execution = (; b = st.execution.b, a = st.execution.a), experiments = (; b = st.experiments.b, a = st.experiments.a)))
    @test residual(model, ps, reordered) ≈ expected(q)
    reordered_model = CorleoneDynamicOptProblem(shooting, experiments; ps, st = reordered)
    @test (@inferred residual(reordered_model)) ≈ expected(q)
    opt = OptimizationProblem(model, SciMLBase.ADTypes.AutoForwardDiff())
    @test opt.f.f(q, opt.p) ≈ objective(q)
    @test opt.f.f(q, nothing) ≈ objective(q)

    inplace(res, trajectory) = (res[1] = last(trajectory.u)[1]; nothing)
    constrained = CorleoneDynamicOptProblem(
        shooting, experiments, :a => :(u₁(1.75) - p₂(1.25)), :b => inplace,
        trajectories -> last(trajectories.b.u)[1] - last(trajectories.a.u)[1];
        ps, st, lcons = [-Inf, 0.0, -Inf], ucons = [10.0, 20.0, 20.0]
    )
    @test 1.75 in constrained.evaluation_grid
    @test all(t -> t in constrained.evaluation_grid, samples)
    @test !(1.75 in constrained.st.execution.a.timestops)
    @test optimization_constraints(constrained) ≈ [1.5, 6.0, 3.0]
    buffer = zeros(3)
    @test optimization_constraints!(buffer, constrained) === buffer
    @test buffer ≈ optimization_constraints(constrained)
    @test_throws ArgumentError CorleoneDynamicOptProblem(shooting, experiments, :(u₁(1.0)); ps, st, lcons = [0.0], ucons = [1.0])
    @test_throws ArgumentError CorleoneDynamicOptProblem(shooting, experiments, :unknown => :(u₁(1.0)); ps, st, lcons = [0.0], ucons = [1.0])
    changed = merge(st, (; execution = merge(st.execution, (; a = merge(st.execution.a, (; timestops = [0.0, 2.0]))))))
    @test_throws ArgumentError optimization_objective(model, ps, changed)
end

@testset "Observation and constraint conventions at resets" begin
    problem = ODEProblem((u, p, t) -> [p[1]], [0.0], (0.0, 2.0), [1.0])
    experiments = Experiments(
        problem, Tunable(1) => nothing; experiments = (
            run = (
                annotations = (Constant(1; kind = Val(:variable), default = 2.0) => [1.0],),
                observations = (:(u₁ + 0t) => ([1.0], [1.0]),),
            ),
        )
    )
    model = CorleoneDynamicOptProblem(ShootingLayer(problem, Tsit5()), experiments, :run => :(u₁(1.0)); lcons = [2.0], ucons = [2.0])
    @test residual(model) ≈ [0.0] atol = 1.0e-12
    @test optimization_constraints(model) ≈ [2.0]
end

@testset "Reset flags are not part of the schedule" begin
    problem = ODEProblem((u, p, t) -> p, [0.0], (0.0, 1.0), [1.0])
    data = Experiment(1 => ([0.25, 1.0], [0.25, 0.5], 0.1))
    experiments = Experiments(problem, Tunable(1; kind = Val(:variable)) => [0.0, 0.5]; experiments = (a = data,))
    ps, st = LuxCore.setup(Xoshiro(1), experiments)
    model = CorleoneDynamicOptProblem(ShootingLayer(problem, Tsit5()), experiments; ps, st)
    # A fresh state has all reset flags cleared; construction ran a forward pass.
    _, fresh = LuxCore.setup(Xoshiro(1), experiments)
    @test optimization_objective(model, ps, fresh) ≈ optimization_objective(model)
end

@testset "Shared priors contribute once" begin
    problem = ODEProblem((u, p, t) -> p, [0.0], (0.0, 1.0), [1.0])
    prior = Normal(0.0, 2.0)
    data = Experiment(1 => ([1.0], [1.0], 0.1))
    experiments = Experiments(problem, Tunable(1; prior) => nothing; experiments = (a = data, b = data))
    model = CorleoneDynamicOptProblem(ShootingLayer(problem, Tsit5()), experiments)
    @test optimization_objective(model) ≈ -logpdf(prior, 1.0)
    @test (@inferred optimization_objective(model, ComponentArray(model.ps))) ≈ -logpdf(prior, 1.0)
    @test isempty(model.lcons)
end
