using Test, Corleone, LuxCore, OrdinaryDiffEqTsit5, ComponentArrays, Random, ForwardDiff
using Optimization, OptimizationMOI, Ipopt, SparseMatrixColorings
import DifferentiationInterface as DI

@testset "Prepared sparse Jacobian and optimization bridge" begin
    prob = ODEProblem((u, p, t) -> [p[1] * u[1] + p[2]], [1.0], (0.0, 2.0), [0.2, 0.3]; abstol = 1.0e-10, reltol = 1.0e-10)
    layer = ShootingLayer(
        prob, Tsit5(), Tunable(1; default = 0.2, lowerbound = -1.0, upperbound = 1.0) => [0.5, 1.5],
        Constant(2; default = 0.3) => [0.5, 1.5],
        Tunable(1; kind = Val(:variable), default = 1.0) => [1.0]
    )
    ps, st = LuxCore.setup(MersenneTwister(1), layer)
    ps, st = add_parameter_segments!(layer, ps, st)
    model = CorleoneDynamicOptProblem(layer, traj -> (last(traj.u)[1] - 1.0)^2; ps, st)
    sparse_opt = OptimizationProblem(model, AutoForwardDiff(); sparse_constraints = true)
    @test sparse_opt.f.cons_j !== nothing
    @test sparse_opt.f.cons_jac_prototype !== nothing
    q, rebuild = pack(default_vectorizer(), ps)
    changed = deepcopy(st)
    changed.parameter.parameters.p₁.parameter.p₂.parameters.ps .= 0.6
    jacobians = map((st, changed)) do state
        J = copy(sparse_opt.f.cons_jac_prototype)
        sparse_opt.f.cons_j(J, q, state)
        expected = ForwardDiff.jacobian(x -> optimization_constraints(model, rebuild(x), state), q)
        @test Matrix(J) ≈ expected
        J
    end
    @test jacobians[1] != jacobians[2] # DI context must use the current fixed values.
    f = OptimizationFunction(model, AutoForwardDiff(); sparse_constraints = true)
    J = copy(f.cons_jac_prototype)
    f.cons_j(J, q, st)
    @test J ≈ jacobians[1]
    @test_throws ArgumentError OptimizationProblem(model, AutoForwardDiff(); sparse_constraints = true, inplace = false)
    @test_throws ArgumentError OptimizationProblem(model, AutoForwardDiff(); sparse_constraints = true, function_kwargs = (; cons_j = (args...) -> nothing))
    @test_throws ArgumentError OptimizationFunction(model, AutoForwardDiff(); sparse_constraints = true, inplace = false)
    user_constraint! = (res, traj) -> begin
        res[1] = first(traj.u)[1]
        res[2] = last(traj.u)[1]
        nothing
    end
    user_model = CorleoneDynamicOptProblem(layer, traj -> last(traj.u)[1], user_constraint!; ps, st, lcons = [-Inf, -Inf], ucons = [Inf, Inf])
    user_opt = OptimizationProblem(user_model, AutoForwardDiff(); sparse_constraints = true)
    user_J = copy(user_opt.f.cons_jac_prototype)
    user_opt.f.cons_j(user_J, q, st)
    @test Matrix(user_J) ≈ ForwardDiff.jacobian(x -> optimization_constraints(user_model, rebuild(x), st), q)
    parsed_model = CorleoneDynamicOptProblem(layer, :(u₁(2.0)), :([u₁(1.0), p₂(1.0), p₁(0.5)]); ps, st, lcons = [-Inf, -Inf, -Inf], ucons = [Inf, Inf, Inf])
    parsed_opt = OptimizationProblem(parsed_model, AutoForwardDiff(); sparse_constraints = true)
    parsed_J = copy(parsed_opt.f.cons_jac_prototype)
    parsed_opt.f.cons_j(parsed_J, q, st)
    @test Matrix(parsed_J) ≈ ForwardDiff.jacobian(x -> optimization_constraints(parsed_model, rebuild(x), st), q)
    @test count(constraint_jacobian_sparsity(parsed_model)[1:3, :]) < 3length(q)
    unconstrained = CorleoneDynamicOptProblem(ShootingLayer(prob, Tsit5()), traj -> last(traj.u)[1])
    empty_opt = OptimizationProblem(unconstrained, AutoForwardDiff(); sparse_constraints = true)
    @test empty_opt.f.cons === nothing
    @test empty_opt.f.cons_j === nothing
    outcomes = map((false, true)) do sparse_constraints
        opt = OptimizationProblem(model, AutoForwardDiff(); sparse_constraints)
        sol = solve(opt, Ipopt.Optimizer(); tol = 1.0e-7, max_iter = 200, hessian_approximation = "limited-memory", print_level = 0)
        @test successful_retcode(sol)
        @test maximum(abs, optimization_constraints(model, rebuild(sol.u))) < 1.0e-6
        sol.objective
    end
    @test outcomes[1] ≈ outcomes[2] atol = 1.0e-7
end
