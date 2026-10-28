using Test
using Corleone
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using OrdinaryDiffEqTsit5
using ComponentArrays, ForwardDiff
using Optimization, OptimizationMOI, Ipopt
using SymbolicIndexingInterface
using LuxCore, Random

@variables x(..) = 1.0 [tunable = false]
@variables u(..) = 0.0 [bounds = (-1.0, 1.0), input = true]
@parameters a = 2.0 [bounds = (1.0, 3.0)] b = 3.0 [tunable = false]
I = ModelingToolkit.Symbolics.Integral(t in (0.0, 1.0))
@named system = System(
    [D(x(t)) ~ a + u(t) + b - 3], t;
    costs = [I(x(t)^2), x(1.0)^2],
    constraints = [x(0.5) ≳ 1.5, a ~ 2.0, u(0.25) ≲ 0.5]
)

function make_model(; kwargs...)
    return CorleoneDynamicOptProblem(
        system, [], u(t) => 0.0:0.25:0.75;
        algorithm = Tsit5(), problem_kwargs = (; abstol = 1.0e-10, reltol = 1.0e-10), kwargs...
    )
end

@testset "MTK annotations" begin
    compiled = mtkcompile(system; inputs = [u(t)], split = false)
    prob = ODEProblem(compiled, [x(t) => 1.25, a => 2.5], (0.0, 1.0); build_initializeprob = false, check_compatibility = false)
    annotations = Annotations(compiled; u0 = prob.u0, p = prob.p)
    @test all(Corleone.is_fully_specified, vcat(annotations.variables, annotations.parameters))
    state = only(annotations.variables)
    @test !Corleone.is_tunable(state)
    @test state.default == 1.25
    @test state.kind == Val(:variable)
    @test state.index == variable_index(compiled, state.sym)
    parameter = only(filter(annotation -> isequal(annotation.sym, ModelingToolkit.Symbolics.unwrap(a)), annotations.parameters))
    @test parameter.default == 2.5
    @test (parameter.lowerbound, parameter.upperbound) == (1.0, 3.0)
    @test Corleone.is_tunable(parameter)
    @test parameter.kind == Val(:parameter)
    fixed = only(filter(annotation -> isequal(annotation.sym, ModelingToolkit.Symbolics.unwrap(b)), annotations.parameters))
    @test !Corleone.is_tunable(fixed)
    @test fixed.default == 3.0
    @test !fixed.integer
    @test Annotations(compiled).variables[1].default == 1.0
    @test all(Corleone.is_tunable, Annotations(compiled; lift_states = true).variables)
    definitions = map(annotations.parameters) do annotation
        grid = isequal(annotation.sym, ModelingToolkit.Symbolics.unwrap(u(t))) ? [0.5] : nothing
        annotation => grid
    end
    layer = ShootingLayer(prob, Tsit5(), definitions...)
    ps, st = LuxCore.setup(Random.default_rng(), layer)
    @test ps.parameter.parameters[Symbol(u(t))].parameter.ps == zeros(2)
    @test st.parameter.parameters[Symbol(b)].parameters.parameter.ps == [3.0]
end

@testset "MTK frontend" begin
    @test !isnothing(Base.get_extension(Corleone, :CorleoneModelingToolkitExtension))
    model = make_model()
    @test model.layer isa ShootingLayer
    @test model.layer.problem.tspan == (0.0, 1.0)
    @test model.layer.problem.p isa Vector
    @test isnothing(model.layer.problem.f.initialization_data)
    @test optimization_objective(model, model.ps, model.st) ≈ 40 / 3
    @test optimization_constraints(model, model.ps, model.st) ≈ [-0.5, 0.0, -0.5]
    @test model.lcons == [-Inf, 0.0, -Inf]
    @test model.ucons == zeros(3)
    @test model.timepoints[Symbol(x(t))] == [0.5, 1.0]
    @test length(Corleone.control_symbols(model.layer.sys)) == 1
    @test isequal(only(Corleone.control_symbols(model.layer.sys)), ModelingToolkit.Symbolics.unwrap(u(t)))
    @test length(Corleone.quadrature_symbols(model.layer.sys)) == 1
    @test length(variable_symbols(model.layer.sys)) == 2
    @test length(model.ps.parameter.parameters[Symbol(u(t))].parameter.ps) == 4
    @test isempty(model.ps.parameter.parameters[Symbol(b)])
    traj, _ = model.layer((model.layer.problem, model.evaluation_grid), model.ps, model.st)
    @test traj[x(t)][end] ≈ 3.0
    @test traj[u(t)] == zeros(4)

    opt = OptimizationProblem(model, AutoForwardDiff())
    @test opt.f(opt.u0, opt.p) ≈ 40 / 3
    @test length(opt.u0) == 5
    @test count(==(-1.0), opt.lb) == 4
    @test count(==(1.0), opt.ub) == 4
    reconstruct = last(pack(Val(:ComponentArrays), model.ps))
    loss(q) = optimization_objective(model, reconstruct(q), model.st)
    grad = ForwardDiff.gradient(loss, opt.u0)
    finite = map(eachindex(opt.u0)) do i
        left, right = copy(opt.u0), copy(opt.u0)
        left[i] -= 1.0e-5
        right[i] += 1.0e-5
        (loss(right) - loss(left)) / 2.0e-5
    end
    @test grad ≈ finite rtol = 1.0e-6
    residual(q) = optimization_constraints(model, reconstruct(q), model.st)
    @test all(isfinite, ForwardDiff.jacobian(residual, opt.u0))
    result = zeros(3)
    opt.f.cons(result, opt.u0, opt.p)
    @test result ≈ residual(opt.u0)

    # The symbolic inequality/equality bounds are passed straight to SciMLBase.
    solution = solve(opt, Ipopt.Optimizer(); max_iter = 100, tol = 1.0e-7, hessian_approximation = "limited-memory", print_level = 0)
    @test SciMLBase.successful_retcode(solution)
    @test solution.objective < loss(opt.u0)
    @test all(model.lcons .- 1.0e-6 .<= residual(solution.u) .<= model.ucons .+ 1.0e-6)
end

@testset "MTK shooting and evaluator options" begin
    model = make_model(shooting = [0.0, 0.4, 0.8], evaluator = Val(:Eval))
    @test model.nshooting == 4 # Two states at two interior resets.
    @test length(optimization_constraints(model, model.ps, model.st)) == 7
    @test model.lcons[4:end] == model.ucons[4:end] == zeros(4)
    @test all(isfinite, optimization_constraints(model, model.ps, model.st))
    opt = OptimizationProblem(model, AutoForwardDiff())
    @test all(isfinite, ForwardDiff.gradient(q -> opt.f(q, opt.p), opt.u0))

    explicit = make_model(tspan = (0.0, 2.0))
    @test explicit.layer.problem.tspan == (0.0, 2.0)
    @test optimization_objective(explicit, explicit.ps, explicit.st) ≈ 40 / 3
    @test_throws ArgumentError make_model(tspan = (0.0, 0.5))
    @test_throws ArgumentError CorleoneDynamicOptProblem(system, [], u(t) => [0.0, 0.5, 0.25]; algorithm = Tsit5())

    initialized = CorleoneDynamicOptProblem(
        system, [x(t) => 2.0], u(t) => [0.25, 0.5, 0.75]; algorithm = Tsit5(), tspan = (0.0, 1.0)
    )
    @test optimization_objective(initialized, initialized.ps, initialized.st) ≈ 76 / 3

    partial_integral = ModelingToolkit.Symbolics.Integral(t in (0.25, 0.75))
    @named partial_system = System([D(x(t)) ~ a + u(t)], t; costs = [partial_integral(x(t)^2)])
    partial = CorleoneDynamicOptProblem(partial_system, [], u(t) => [0.0, 0.5]; algorithm = Tsit5(), tspan = (0.0, 1.0))
    @test optimization_objective(partial, partial.ps, partial.st) ≈ (2.5^3 - 1.5^3) / 6

    @parameters array_parameter[1:1] = [1.0]
    @named array_system = System([D(x(t)) ~ array_parameter[1] + u(t)], t; costs = [x(1.0)])
    @test_throws ArgumentError CorleoneDynamicOptProblem(array_system, [], u(t) => [0.0, 0.5]; algorithm = Tsit5(), tspan = (0.0, 1.0))
end

@testset "Nonlinear MTK construction" begin
    @variables prey(..) = 0.5 [tunable = false] predator(..) = 0.7 [tunable = false]
    @variables fishing(..) = 0.0 [input = true, bounds = (0.0, 1.0)]
    @constants catch_prey = 0.4 catch_predator = 0.2
    integral = ModelingToolkit.Symbolics.Integral(t in (0.0, 12.0))
    @named lotka = System(
        [
            D(prey(t)) ~ prey(t) - prey(t) * predator(t) - catch_prey * fishing(t) * prey(t),
            D(predator(t)) ~ -predator(t) + prey(t) * predator(t) - catch_predator * fishing(t) * predator(t),
        ], t; costs = [integral((prey(t) - 1)^2 + (predator(t) - 1)^2)]
    )
    model = CorleoneDynamicOptProblem(lotka, [], fishing(t) => 0.0:0.1:11.9; algorithm = Tsit5())
    @test optimization_objective(model, model.ps, model.st) ≈ 6.062277381976436 atol = 1.0e-4
    q, reconstruct = pack(Val(:ComponentArrays), model.ps)
    gradient = ForwardDiff.gradient(q -> optimization_objective(model, reconstruct(q), model.st), q)
    @test length(gradient) == 120
    @test all(isfinite, gradient)
end

@testset "Tunable MTK initial values" begin
    @variables free_state(..) = 0.5 [tunable = true, bounds = (0.3, 1.0)]
    @variables fixed_state(..) = 0.7 [tunable = false]
    @parameters rate = 0.0 [tunable = false]
    @named initial_system = System(
        [D(free_state(t)) ~ rate * free_state(t), D(fixed_state(t)) ~ rate * fixed_state(t)],
        t; costs = [free_state(1.0)^2]
    )
    model = CorleoneDynamicOptProblem(initial_system, []; algorithm = Tsit5(), tspan = (0.0, 1.0))
    free_name = Symbol(free_state(t))
    @test model.ps.initialcondition.initialconditions[free_name].parameter.ps == [0.5]
    @test isempty(model.ps.initialcondition.initialconditions[Symbol(fixed_state(t))])
    @test model.nshooting == 0
    @test isempty(model.lcons)
    opt = OptimizationProblem(model, AutoForwardDiff())
    @test opt.u0 == [0.5]
    @test opt.lb == [0.3]
    @test opt.ub == [1.0]
    _, reconstruct = pack(Val(:ComponentArrays), model.ps)
    current = reconstruct([0.8])
    trajectory, _ = model.layer(nothing, current, model.st)
    @test trajectory[free_state(t)] ≈ [0.8, 0.8]
    @test trajectory[fixed_state(t)] ≈ [0.7, 0.7]
    @test isempty(shooting_constraints(trajectory))
    @test optimization_objective(model, current) ≈ 0.64
    @test ForwardDiff.gradient(q -> optimization_objective(model, reconstruct(q)), [0.8]) ≈ [1.6]
    solution = solve(opt, Ipopt.Optimizer(); print_level = 0, tol = 1.0e-8, hessian_approximation = "limited-memory")
    @test SciMLBase.successful_retcode(solution)
    @test only(solution.u) ≈ 0.3 atol = 1.0e-6

    for ensemblealg in (EnsembleSerial(), EnsembleThreads())
        multiple = CorleoneDynamicOptProblem(
            initial_system, []; algorithm = Tsit5(), tspan = (0.0, 1.0),
            shooting = [0.0, 0.5], ensemblealg
        )
        @test multiple.ps.initialcondition.initialconditions[free_name].parameter.ps == [0.5, 0.5]
        @test multiple.nshooting == 2
        ps = deepcopy(multiple.ps)
        ps.initialcondition.initialconditions[free_name].parameter.ps[1] = 0.8
        trajectory, _ = multiple.layer(nothing, ps, multiple.st)
        @test isempty(first(trajectory.residuals))
        @test length(shooting_constraints(trajectory)) == 2
        @test sort(shooting_constraints(trajectory)) ≈ [-0.3, 0.0] atol = 1.0e-12
    end
end
