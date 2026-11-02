using Test, Corleone, LuxCore, OrdinaryDiffEqTsit5, ComponentArrays, Random, ForwardDiff, SparseArrays, LinearAlgebra

@testset "Objective and matching contributions to Lagrangian structure" begin
    problem = ODEProblem((u, p, t) -> [p[1] * u[1] + p[2]], [0.6], (0.0, 1.0), [0.2, 0.1]; abstol = 1.0e-11, reltol = 1.0e-11)
    layer = ShootingLayer(problem, Tsit5(), Tunable(1; kind = Val(:variable), default = 0.6) => [0.0, 0.5], Tunable(1; default = 0.2) => [0.25, 0.75], Constant(2; default = 0.1) => nothing)
    ps, st = LuxCore.setup(MersenneTwister(1), layer)
    ps, st = add_parameter_segments!(layer, ps, st)
    q, rebuild = pack(default_vectorizer(), ps)
    ids = rebuild(collect(eachindex(q)))
    node = ids.initialcondition.initialconditions.u₁.parameter.ps[2]
    controls = ids.parameter.parameters.p₁.parameter.ps
    matching_only = CorleoneDynamicOptProblem(layer, :(0.0 + 0.0); ps, st)
    P = lagrangian_hessian_sparsity(matching_only)
    @test P isa SparseMatrixCSC{Bool}
    @test issymmetric(P)
    @test !any(P[node, :]) # Lifted assignment is affine in the matching residual.
    @test !any(P[controls[3], :]) # New parameter occurs only in an affine equality.
    objective = CorleoneDynamicOptProblem(layer, :(u₁(1.0)^2); ps, st)
    cross = CorleoneDynamicOptProblem(layer, :(u₁(1.0)^2), :(u₁(0.5) * p₁(0.25)); ps, st, lcons = [0.0], ucons = [Inf])
    @test lagrangian_hessian_sparsity(objective)[node, node]
    @test !lagrangian_hessian_sparsity(objective)[controls[2], controls[3]]
    @test lagrangian_hessian_sparsity(cross)[node, controls[2]]
    for model in (matching_only, objective, cross)
        pattern = lagrangian_hessian_sparsity(model)
        @test issymmetric(pattern)
        for shift in (0.0, 0.02), sigma in (0.0, 1.0)
            input = q .+ shift
            weights = [(-1.0)^i / i for i in eachindex(model.lcons)]
            H = ForwardDiff.hessian(input) do x
                sigma * optimization_objective(model, rebuild(x), st) + dot(weights, optimization_constraints(model, rebuild(x), st))
            end
            @test maximum(abs, H[.!Matrix(pattern)]; init = 0.0) < 1.0e-8
        end
    end
    opaque = CorleoneDynamicOptProblem(layer, traj -> last(traj.u)[1]^2; ps, st)
    @test all(lagrangian_hessian_sparsity(opaque))
end

@testset "Unconstrained objective and empty Hessian structure" begin
    problem = ODEProblem((u, p, t) -> p, [1.0], (0.0, 1.0), [0.2])
    layer = ShootingLayer(problem, Tsit5(), Tunable(1; default = 0.2) => [0.25, 0.75])
    model = CorleoneDynamicOptProblem(layer, :(p₁(0.5)^2))
    @test isempty(model.lcons)
    q, rebuild = pack(default_vectorizer(), model.ps)
    pattern = lagrangian_hessian_sparsity(model)
    @test findall(pattern) == [CartesianIndex(2, 2)]
    H = ForwardDiff.hessian(x -> optimization_objective(model, rebuild(x)), q)
    @test H ≈ 2Matrix(pattern)
    fixed = ShootingLayer(problem, Tsit5(); fixed_parameters = [1])
    model = CorleoneDynamicOptProblem(fixed, :(0.0 + 0.0))
    @test size(lagrangian_hessian_sparsity(model)) == (0, 0)
end

@testset "Partial and fixed resets in Hessian dependencies" begin
    problem = ODEProblem((u, p, t) -> [p[1] * u[2], u[1]^2], [0.5, 0.4], (0.0, 1.0), [0.2]; abstol = 1.0e-11, reltol = 1.0e-11)
    layer = ShootingLayer(problem, Tsit5(), Tunable(1; kind = Val(:variable), default = 0.5) => [0.5], Constant(2; kind = Val(:variable), default = 0.4) => [0.75], Tunable(1; default = 0.2) => [0.25, 0.75])
    model = CorleoneDynamicOptProblem(layer, :(0.0 + 0.0), :(u₂(0.5)^2); lcons = [0.0], ucons = [Inf])
    q, rebuild = pack(default_vectorizer(), model.ps)
    ids = rebuild(collect(eachindex(q)))
    pattern = lagrangian_hessian_sparsity(model)
    @test !any(pattern[ids.initialcondition.initialconditions.u₁.parameter.ps[1], :])
    for shift in (0.0, 0.02)
        H = ForwardDiff.hessian(x -> sum(optimization_constraints(model, rebuild(x))), q .+ shift)
        @test maximum(abs, H[.!Matrix(pattern)]; init = 0.0) < 1.0e-8
    end
end

@testset "General nonlinear expressions use dependency blocks" begin
    problem = ODEProblem((u, p, t) -> p, [1.0], (0.0, 1.0), [0.2])
    layer = ShootingLayer(problem, Tsit5(), Tunable(1; default = 0.2) => [0.25, 0.75])
    for expression in (:(sin(p₁(0.0) + p₁(1.0))), :(p₁(0.0) / p₁(1.0)), :(p₁(0.0) + p₁(1.0)))
        expected = Bool[1 0 1; 0 0 0; 1 0 1]
        objective = CorleoneDynamicOptProblem(layer, expression)
        @test Matrix(lagrangian_hessian_sparsity(objective)) == expected
        model = CorleoneDynamicOptProblem(layer, :(0 + 0), expression; lcons = [0.0], ucons = [Inf])
        pattern = lagrangian_hessian_sparsity(model)
        @test Matrix(pattern) == expected
        q, rebuild = pack(default_vectorizer(), model.ps)
        for shift in (0.0, 0.1)
            H = ForwardDiff.hessian(x -> sum(optimization_constraints(model, rebuild(x))), q .+ shift)
            @test maximum(abs, H[.!Matrix(pattern)]; init = 0.0) < 1.0e-8
        end
    end
end
