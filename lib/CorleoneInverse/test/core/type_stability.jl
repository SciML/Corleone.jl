using Test, CorleoneInverse, LuxCore, Random, ComponentArrays, OrdinaryDiffEqTsit5, ForwardDiff, Zygote

# Experiments share rates and an observation gain; each applies its own constant input.
function stability_model(; heterogeneous)
    rhs(u, p, t) = [p[1] * u[1] - p[2] * u[1] * u[2] + p[4], -p[3] * u[2] + p[2] * u[1] * u[2]]
    problem = ODEProblem(rhs, [1.0, 0.5], (0.0, 2.0), [1.0, 0.8, 0.6, 0.0]; abstol = 1.0e-9, reltol = 1.0e-9)
    t = collect(0.25:0.25:2.0)
    observations = (:u₁ => (t, fill(1.0, length(t)), 0.1), :(w * u₂) => (t, fill(0.5, length(t)), 0.1))
    spec(input) = (; annotations = (Constant(4; default = input) => nothing,), observations)
    specs = (a = spec(0.0), b = spec(0.1), c = spec(0.2))
    if heterogeneous
        # A different annotation set and observation tuple give this experiment its own types.
        local_initial = Tunable(1; kind = Val(:variable), default = 1.2) => [0.0]
        mixed = (; annotations = (Constant(4; default = 0.3) => nothing, local_initial), observations = (:u₁ => (t, fill(1.1, length(t)), 0.2),))
        specs = merge(specs, (; d = mixed))
    end
    rates = Tunable([1, 2, 3]; default = [1.0, 0.8, 0.6]) => nothing
    experiments = Experiments(problem, rates; controls = [4], experiments = specs)
    shooting = ShootingLayer(problem, Tsit5())
    return CorleoneDynamicOptProblem(shooting, experiments; observation_parameters = (w = 1.0,), reducer = sum)
end

@testset "Inferable experiment pipeline ($(heterogeneous ? "heterogeneous" : "homogeneous"))" for heterogeneous in (false, true)
    model = stability_model(; heterogeneous)
    ps, st = model.ps, model.st
    trajectories = @inferred Corleone.optimization_trajectory(model, ps, st)
    @test trajectories isa NamedTuple
    @test all(isconcretetype ∘ typeof, values(trajectories))
    # Experiments with equal annotations and observations share one trajectory type.
    @test allequal(typeof.(values(trajectories))) == !heterogeneous
    @test (@inferred optimization_objective(model, ps)) isa Float64
    @test (@inferred residual(model, ps)) isa Vector{Float64}

    q, reconstruct = pack(default_vectorizer(), ps)
    @test (@inferred optimization_objective(model, reconstruct(q))) ≈ optimization_objective(model, ps)
    loss = let model = model, reconstruct = reconstruct
        x -> optimization_objective(model, reconstruct(x))
    end
    dual = ForwardDiff.Dual{Nothing}.(q, Ref(ForwardDiff.Partials((1.0,))))
    @test (@inferred loss(dual)) isa ForwardDiff.Dual{Nothing, Float64, 1}
    # A fixed configuration fixes the chunk size, which `gradient(f, x)` picks at runtime.
    config = ForwardDiff.GradientConfig(loss, q, ForwardDiff.Chunk{length(q)}())
    gradient = @inferred ForwardDiff.gradient(loss, q, config)
    @test gradient isa Vector{Float64}
    @test all(isfinite, gradient)
end

@testset "Reverse-mode residual assembly ($(heterogeneous ? "heterogeneous" : "homogeneous"))" for heterogeneous in (false, true)
    model = stability_model(; heterogeneous)
    trajectories = Corleone.optimization_trajectory(model, model.ps, model.st)
    data = model.objective.data
    # Differentiate the assembly alone; equally typed experiments share the mapped path.
    loss(w) = sum(abs2, CorleoneInverse.measurement_residuals(data, trajectories, (only(w),)))
    w = [0.7]
    expected = ForwardDiff.gradient(loss, w)
    gradient = only(Zygote.gradient(loss, w))
    @test gradient ≈ expected
    @test any(!iszero, expected)
end
