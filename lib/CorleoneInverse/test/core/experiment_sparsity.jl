using Test, CorleoneInverse, ComponentArrays, LuxCore, OrdinaryDiffEqTsit5, ForwardDiff, Random
import DifferentiationInterface, SparseMatrixColorings
using TransformVariables: asℝ₊

@testset "Experiment evaluation and packed sparsity" begin
    problem = ODEProblem((u, p, t) -> [p[1]], [0.0], (0.0, 1.0), [2.0])
    shooting = ShootingLayer(problem, Tsit5(), Tunable(1) => nothing)
    times = [0.25, 0.75]
    data = Experiment(:(u₁^2 - w) => (times, [0.5, 1.0], 0.5))
    specification(x) = (;
        annotations = Tunable(1; kind = Val(:variable), default = x) => [0.0],
        observations = data,
    )
    experiments = Experiments(shooting, (a = specification(1.0), b = specification(2.0)))
    shooting_ps, shooting_st = LuxCore.setup(Random.default_rng(), experiments)
    model = CorleoneDynamicOptProblem(
        shooting, experiments, :a => :(u₁(0.25)), :b => :(u₁(0.75));
        ps = shooting_ps, st = shooting_st, observation_parameters = (w = 0.5,),
        lcons = (-Inf, -Inf), ucons = (Inf, Inf)
    )
    layer = model.layer.evaluation
    state = (; shooting = deepcopy(shooting_st), evaluation = (; marker = true))
    trajectories, next = layer(experiments, model.ps, state)
    expected, expected_st = shooting(experiments, shooting_ps, deepcopy(shooting_st))
    @test getproperty.(trajectories, :t) == getproperty.(expected, :t)
    @test getproperty.(trajectories, :u) ≈ getproperty.(expected, :u)
    @test next.evaluation == state.evaluation
    @test next.shooting.execution == expected_st.execution
    updated_ps = merge(model.ps, (; evaluation = (w = 1.5,)))
    updated, _ = layer(experiments, updated_ps, deepcopy(state))
    @test getproperty.(updated, :u) ≈ getproperty.(trajectories, :u)
    @test residual(model, updated_ps) ≈ residual(model) .+ 2

    ps = (; model.ps.evaluation, model.ps.shooting)
    q, reconstruct = pack(default_vectorizer(), ps)
    # Reversed root packing gives shared observation parameter first.
    @test q == [0.5, 2.0, 1.0, 2.0]
    J = Matrix(residual_jacobian_sparsity(model; ps))
    C = Matrix(constraint_jacobian_sparsity(model; ps))
    H = Matrix(lagrangian_hessian_sparsity(model; ps))
    @test J == Bool[1 1 1 0; 1 1 1 0; 1 1 0 1; 1 1 0 1]
    @test C == Bool[0 1 1 0; 0 1 0 1]
    @test H == Bool[1 1 1 1; 1 1 1 1; 1 1 1 0; 1 1 0 1]

    ad = Corleone.SciMLBase.ADTypes.AutoForwardDiff()
    opt = Corleone.SciMLBase.OptimizationProblem(model, ad; ps, sparse_constraints = true, sparse_hessian = true)
    for x in (q, q .+ 0.2, [2.0, 0.0, 1.0, 1.0]), λ in ([1.0, 0.0], [0.5, -1.5])
        r(q) = residual(model, reconstruct(q))
        c(q) = optimization_constraints(model, reconstruct(q))
        ℓ(q) = Corleone.optimization_lagrangian(model, reconstruct(q), model.st, 1.0, λ)
        dense_r = ForwardDiff.jacobian(r, x)
        dense_c = ForwardDiff.jacobian(c, x)
        dense_h = ForwardDiff.hessian(ℓ, x)
        @test all(iszero, dense_r[.!J])
        @test all(iszero, dense_c[.!C])
        @test all(iszero, dense_h[.!H])
        jacobian = copy(opt.f.cons_jac_prototype)
        hessian = copy(opt.f.lag_hess_prototype)
        opt.f.cons_j(jacobian, x, model.st)
        opt.f.lag_h(hessian, x, 1.0, λ, model.st)
        @test Matrix(jacobian) ≈ dense_c
        @test Matrix(hessian) ≈ dense_h
    end

    coupled_loss = CorleoneDynamicOptProblem(
        shooting, experiments; observation_parameters = (w = 0.5,), reducer = r -> sum(r)^2
    )
    @test lagrangian_hessian_sparsity(coupled_loss)[2, 3]
    coupled_constraint = trajectories -> first(trajectories.a.u)[1] * first(trajectories.b.u)[1]
    coupled_model = CorleoneDynamicOptProblem(
        shooting, experiments, coupled_constraint; observation_parameters = (w = 0.5,), lcons = 0.0, ucons = Inf
    )
    @test lagrangian_hessian_sparsity(coupled_model)[2, 3]

    appended = append_experiment(
        model, opt, :c => specification(3.0); ps, u0 = [1.5, 2.0, 1.0, 2.0],
        sparse_constraints = true, sparse_hessian = true
    )
    @test appended.model.ps.evaluation == (w = 1.5,)
    @test size(residual_jacobian_sparsity(appended.model)) == (6, 5)
    @test size(appended.problem.f.cons_jac_prototype) == (2, 5)
    @test size(appended.problem.f.lag_hess_prototype) == (5, 5)
end

@testset "Experiment sample sides and affine matching" begin
    problem = ODEProblem((du, u, p, t) -> (du[1] = p[1]; nothing), [0.0], (0.0, 2.0), [2.0])
    shooting = ShootingLayer(
        problem, Tsit5(), Constant(1; default = 2.0) => nothing,
        Tunable(1; kind = Val(:variable), default = [1.0, 5.0]) => [0.0, 1.0]
    )
    data = Experiment(:(u₁ - w) => ([1.0, 1.5], zeros(2)))
    model = CorleoneDynamicOptProblem(shooting, data, :(u₁(1.0)); observation_parameters = (w = 0.0,), lcons = 0.0, ucons = Inf)
    q, reconstruct = pack(default_vectorizer(), model.ps)
    @test q == [1.0, 5.0, 0.0]
    @test Matrix(residual_jacobian_sparsity(model)) == Bool[1 0 1; 0 1 1]
    @test Matrix(constraint_jacobian_sparsity(model)) == Bool[0 1 0; 1 1 0]
    @test ForwardDiff.jacobian(q -> residual(model, reconstruct(q)), q) ≈ [-1 0 1; 0 -1 1]
    @test !lagrangian_hessian_sparsity(model)[1, 2]

    lifted = ShootingLayer(
        problem, Tsit5(), Tunable(1; default = 2.0) => nothing,
        Tunable(1; kind = Val(:variable), default = [1.0, 5.0]) => [0.0, 1.0]
    )
    ps, st = LuxCore.setup(Random.default_rng(), lifted)
    ps, st = add_parameter_segments!(lifted, ps, st)
    only_observation_parameter = Experiment(:(w + 0.0) => ([1.5], [0.0]))
    matching = CorleoneDynamicOptProblem(lifted, only_observation_parameter; ps, st, observation_parameters = (w = 0.0,))
    q, reconstruct = pack(default_vectorizer(), matching.ps)
    ids = reconstruct(collect(eachindex(q)))
    controls = ids.shooting.parameter.parameters.p₁.parameter.ps
    @test findall(!iszero, constraint_jacobian_sparsity(matching)[end, :]) == controls
    @test !any(lagrangian_hessian_sparsity(matching)[last(controls), :])

    transformed = ShootingLayer(
        problem, Tsit5(), Constant(1; default = 2.0) => nothing,
        Tunable(1; kind = Val(:variable), default = [1.0, 5.0], transform = asℝ₊) => [0.0, 1.0]
    )
    transformed_model = CorleoneDynamicOptProblem(transformed, only_observation_parameter; observation_parameters = (w = 0.0,))
    q, reconstruct = pack(default_vectorizer(), transformed_model.ps)
    pattern = lagrangian_hessian_sparsity(transformed_model)
    hessian = ForwardDiff.hessian(q -> sum(optimization_constraints(transformed_model, reconstruct(q))), q)
    @test hessian[2, 2] ≈ 5.0
    @test all(iszero, hessian[.!Matrix(pattern)])
end
