using Test, Corleone, ComponentArrays, OrdinaryDiffEqTsit5, SparseArrays
import OptimizationBase

@testset "OptimizationBase canonical Lagrangian coordinates" begin
    problem = ODEProblem((u, p, t) -> p, [1.0], (0.0, 1.0), [0.2])
    layer = ShootingLayer(problem, Tsit5(), Tunable(1; default = 0.2) => [0.25, 0.75])
    model = CorleoneDynamicOptProblem(layer, :((p₁(0.0) + p₁(1.0))^2))
    prototype = lagrangian_hessian_sparsity(model)
    rows, cols = OptimizationBase.lag_hess_structure(model)
    @test (rows, cols) == OptimizationBase.lag_hess_structure(prototype)
    @test collect(zip(rows, cols)) == [(1, 1), (1, 3), (3, 3)]
    @test all(rows .<= cols)
    @test issorted(collect(zip(cols, rows))) # CSC column order, not row order.
    parameters = ComponentArray(model.ps)
    @test OptimizationBase.lag_hess_structure(model; ps = parameters) == (rows, cols)
end

using Optimization, OptimizationMOI, Ipopt, ForwardDiff, LinearAlgebra
import DifferentiationInterface, SparseMatrixColorings

@testset "Prepared sparse Lagrangian Hessians" begin
    problem = ODEProblem((u, p, t) -> [p[1] * u[1] + p[2]], [0.6], (0.0, 1.0), [0.2, 0.1]; abstol = 1.0e-11, reltol = 1.0e-11)
    layer = ShootingLayer(
        problem, Tsit5(), Tunable(1; default = 0.2, lowerbound = -1.0, upperbound = 1.0) => [0.25, 0.75],
        Constant(2; default = 0.1) => nothing, Tunable(1; kind = Val(:variable), default = 0.6) => [0.5]
    )
    model = CorleoneDynamicOptProblem(layer, :((u₁(1.0) - 1)^2), :(sin(p₁(0.0)) + p₁(1.0)^2); lcons = [-Inf], ucons = [Inf])
    opt = OptimizationProblem(model, AutoForwardDiff(); sparse_constraints = true, sparse_hessian = true)
    q, rebuild = pack(default_vectorizer(), model.ps)
    rows, cols = OptimizationBase.lag_hess_structure(opt.f.lag_hess_prototype)
    changed = deepcopy(opt.p)
    changed.parameter.parameters.p₂.parameters.parameter.ps .= 0.3
    expected_matrices = Matrix[]
    for state in (opt.p, changed), sigma in (0.0, 1.0, 2.0), shift in (0.0, 0.02)
        x = q .+ shift
        lambda = collect(range(-0.5, 0.7; length = length(model.lcons)))
        expected = ForwardDiff.hessian(x) do z
            sigma * optimization_objective(model, rebuild(z), state) + dot(lambda, optimization_constraints(model, rebuild(z), state))
        end
        values = zeros(length(rows))
        opt.f.lag_h(values, x, sigma, lambda, state)
        @test values ≈ [expected[i, j] for (i, j) in zip(rows, cols)] rtol = 1.0e-7 atol = 1.0e-8
        H = copy(opt.f.lag_hess_prototype)
        opt.f.lag_h(H, x, sigma, lambda, state)
        @test Matrix(H) ≈ expected rtol = 1.0e-7 atol = 1.0e-8
        dense_H = fill(NaN, length(x), length(x))
        opt.f.hess(dense_H, x, state)
        objective_H = ForwardDiff.hessian(z -> optimization_objective(model, rebuild(z), state), x)
        @test dense_H ≈ objective_H rtol = 1.0e-7 atol = 1.0e-8
        push!(expected_matrices, expected)
    end
    @test expected_matrices[1] != expected_matrices[7]
    f = OptimizationFunction(model, AutoForwardDiff(); sparse_hessian = true)
    @test f.lag_hess_prototype == opt.f.lag_hess_prototype
    @test_throws ArgumentError OptimizationFunction(model, AutoForwardDiff(); sparse_hessian = true, inplace = false)
    @test_throws ArgumentError OptimizationProblem(model, AutoForwardDiff(); sparse_hessian = true, inplace = false)
    @test_throws ArgumentError OptimizationFunction(model, AutoForwardDiff(); sparse_hessian = true, lag_h = (args...) -> nothing)
    @test_throws ArgumentError OptimizationProblem(model, AutoForwardDiff(); sparse_hessian = true, function_kwargs = (; hess_prototype = zeros(length(q), length(q))))
    sol = solve(opt, Ipopt.Optimizer(); tol = 1.0e-7, max_iter = 200, print_level = 0)
    @test successful_retcode(sol)
    @test maximum(abs, optimization_constraints(model, rebuild(sol.u))[2:end]) < 1.0e-6
end

@testset "Sparse objective Hessian without constraints" begin
    problem = ODEProblem((u, p, t) -> p, [1.0], (0.0, 1.0), [0.2])
    layer = ShootingLayer(problem, Tsit5(), Tunable(1; default = 0.2, lowerbound = -1.0, upperbound = 1.0) => [0.5])
    for expression in (:(p₁(0.0)^2 + p₁(1.0)^2), :(0.0 + 0.0))
        model = CorleoneDynamicOptProblem(layer, expression)
        opt = OptimizationProblem(model, AutoForwardDiff(); sparse_hessian = true)
        H = copy(opt.f.hess_prototype)
        opt.f.hess(H, opt.u0, opt.p)
        _, rebuild = pack(default_vectorizer(), model.ps)
        expected = ForwardDiff.hessian(q -> optimization_objective(model, rebuild(q)), opt.u0)
        @test Matrix(H) ≈ expected
        sol = solve(opt, Ipopt.Optimizer(); tol = 1.0e-7, max_iter = 100, print_level = 0)
        @test successful_retcode(sol)
    end
end
