using OptimalControlBenchmarks, Corleone, Test, ComponentArrays, ForwardDiff, Ipopt
using Optimization, LuxCore, Random, LinearAlgebra
const OCB = OptimalControlBenchmarks

# Exact integral of the piecewise-constant LQR trajectory x'=-x+u, x(0)=1.
function reference_lqr(controls)
    x = one(eltype(controls))
    h = 10.0 / length(controls)
    cost = zero(eltype(controls))
    for u in controls
        cost += 10 * ((u - 3)^2 * h + 2 * (u - 3) * (x - u) * (1 - exp(-h)) + (x - u)^2 * (1 - exp(-2h)) / 2) + 0.1u^2 * h
        x = u + (x - u) * exp(-h)
    end
    return cost
end

@testset "Current optimization bridge and analytic LQR" begin
    q = zeros(4)
    reference = -(ForwardDiff.hessian(reference_lqr, q) \ ForwardDiff.gradient(reference_lqr, q))
    optimum = reference_lqr(reference)
    for shooting in (Float64[], [0.0, 0.5, 1.0])
        grids = BenchmarkGrids(; control_grid = [0.0, 0.25, 0.5, 0.75, 1.0], shooting_grid = shooting, constraint_grid = [0.0, 0.5, 1.0])
        model = prepare_benchmark(OCB.lqr, grids; problem_kwargs = (; abstol = 1.0e-10, reltol = 1.0e-10))
        @test model.layer.problem.tspan == (0.0, 10.0)
        @test length(control_symbols(model.layer.sys)) == 1
        @test get_number_of_shooting_constraints(model.layer, model.ps, model.st) == (isempty(shooting) ? 0 : 2)
        solution = solve_with_corleone(
            OCB.lqr, Ipopt.Optimizer(), grids, false;
            model_kwargs = (; problem_kwargs = (; abstol = 1.0e-10, reltol = 1.0e-10)),
            print_level = 0, tol = 1.0e-9, max_iter = 200
        )
        @test OCB.Optimization.SciMLBase.successful_retcode(solution)
        @test solution.objective ≈ optimum rtol = 1.0e-7
        _, reconstruct = pack(default_vectorizer(), model.ps)
        current = reconstruct(solution.u)
        trajectory, _ = model.layer((model.layer.problem, model.evaluation_grid), current, model.st)
        @test norm(shooting_constraints(trajectory)) < 1.0e-6
        @test OCB.plot_oc_problem(trajectory) isa OCB.CairoMakie.Figure
        @test grids.control_grid == [0.0, 0.25, 0.5, 0.75, 1.0]
    end
end

@testset "Symbolic constraints, metadata, and returned timings" begin
    grids = BenchmarkGrids(; control_grid = [0.0, 0.5], constraint_grid = [0.0, 0.5, 1.0])
    model = prepare_benchmark(OCB.bryson_denham, grids; problem_kwargs = (; abstol = 1.0e-10, reltol = 1.0e-10))
    @test length(model.lcons) == 4 # Two terminal equalities and two path inequalities.
    @test count(iszero, model.lcons) == 2
    @test count(==(-Inf), model.lcons) == 2
    q, reconstruct = pack(default_vectorizer(), model.ps)
    loss(q) = optimization_objective(model, reconstruct(q), model.st)
    @test all(isfinite, ForwardDiff.gradient(loss, q))
    records = run_all([OCB.lqr], Ipopt.Optimizer(), grids; print_level = 0)
    @test length(records) == 1
    @test only(records).name == :lqr
    @test only(records).time > 0
    @test isfinite(only(records).objective)
    @test OCB.Optimization.SciMLBase.successful_retcode(only(records).retcode)
    @test grids.control_grid == [0.0, 0.5]
end

@testset "Rocket nominal trajectory remains finite" begin
    grids = BenchmarkGrids(;
        control_grid = [0.0, 0.25, 0.5, 0.75],
        shooting_grid = [0.0, 0.25, 0.5, 0.75, 1.0], constraint_grid = [0.0, 0.5, 1.0]
    )
    model = prepare_benchmark(OCB.goddarts_rocket, grids)
    @test model.layer.problem.tspan == (0.0, 10.0)
    @test isfinite(optimization_objective(model))
    @test all(isfinite, optimization_constraints(model))
end

@testset "Custom constructors may retain horizon inference" begin
    legacy(grids) = Base.structdiff(OCB.lqr(grids), (; tspan = nothing))
    grids = BenchmarkGrids(; control_grid = [0.0, 0.5])
    @test prepare_benchmark(legacy, grids).layer.problem.tspan == (0.0, 10.0)
end
