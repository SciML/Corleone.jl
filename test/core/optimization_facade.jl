using Test, Corleone, LuxCore, OrdinaryDiffEqTsit5, ComponentArrays, Random, ForwardDiff
using RuntimeGeneratedFunctions
using SymbolicIndexingInterface

facade_helper(x) = 2x + 1

@testset "Timed expressions and structured constraint bounds" begin
    calls = Ref(0)
    initialize = (rng, n::Int) -> (calls[] += 1; fill(0.5, n))
    problem = ODEProblem(
        (u, p, t) -> [p[1]], [1.0], (0.0, 3.0), [0.5];
        saveat = [0.25, 0.75, 1.25, 1.75, 2.25, 2.75], abstol = 1.0e-11, reltol = 1.0e-11
    )
    layer = ShootingLayer(
        problem, Tsit5(),
        Tunable(1; kind = Val(:variable), default = [3.0, 4.0]) => [1.0, 2.0],
        Tunable(1; kind = Val(:parameter), default = initialize) => [1.5]
    )
    original_ps, original_st = LuxCore.setup(MersenneTwister(1), layer)
    ps, st = add_parameter_segments!(layer, original_ps, original_st)
    sides = traj -> [traj.u[searchsortedfirst(traj.t, 1.0)][1], traj.u[searchsortedlast(traj.t, 1.0)][1]]
    inplace! = (res, traj) -> begin
        res[1] = last(traj.u)[1]^2
        res[2] = Corleone.SymbolicIndexingInterface.parameter_values(traj)[1]
        :ignored
    end
    scalar! = (res, traj) -> (res[1] = first(traj.u)[1]; -100)
    model = CorleoneDynamicOptProblem(
        layer, :(u₁(3.0) + p₁(1.5)),
        :(u₁(1.0) + u₁(2.0)), sides,
        :([p₁(0.5), p₁(1.5), u₁(2.25) * p₁(0.75)]), inplace!, scalar!, traj -> last(traj.u)[1];
        ps, st, lcons = [0.0, [0.0, 0.0], zeros(3), zeros(2), 0.0, 0.0],
        ucons = [10.0, [10.0, 10.0], fill(10.0, 3), [100.0, 2.0], 10.0, 10.0]
    )
    @test calls[] == 1
    @test model.ps === ps && model.st === st && model.layer === layer
    @test model.timepoints[:u₁] == [1.0, 2.0, 2.25, 3.0]
    @test model.timepoints[:p₁] == [0.5, 0.75, 1.5]
    @test model.evaluation_grid == sort(unique(vcat(st.timestops, [0.5, 0.75, 2.25])))
    @test all(t -> t in model.times, problem.kwargs[:saveat])
    @test count(==(1.0), model.times) == 2
    @test optimization_objective(model) ≈ 5.0
    expected = [7.0, 1.5, 3.0, 0.5, 0.5, 2.0625, 20.25, 0.5, 1.0, 4.5, 1.5, 0.5, 0.0, 0.0]
    @test optimization_constraints(model) ≈ expected
    @test model.lcons == zeros(14)
    @test model.ucons == [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 100.0, 2.0, 10.0, 10.0, 0.0, 0.0, 0.0, 0.0]
    output = fill(NaN, 14)
    @test optimization_constraints!(output, model) === output
    @test output ≈ expected
    wrong = fill(NaN, 13)
    @test_throws DimensionMismatch optimization_constraints!(wrong, model)
    @test all(isnan, wrong)
    q, rebuild = pack(Val(:ComponentArrays), ps)
    objective, constraints = let model = model, rebuild = rebuild, state = st
        (
            x -> optimization_objective(model, rebuild(x), state),
            x -> optimization_constraints(model, rebuild(x), state),
        )
    end
    @test (@inferred objective(q)) ≈ 5.0
    axes = getaxes(rebuild(q))
    analytic = x -> begin
        tree = ComponentArray(x, axes)
        a, b = tree.initialcondition.initialconditions.u₁.parameter.ps
        p0, p1, p2, p3 = tree.parameter.parameters.p₁.parameter.ps
        final = b + p3
        [
            a + b, 1 + p0, a, p0, p2, (b + 0.25p3) * p0, final^2, p3, 1.0, final,
            a - 1 - p0, b - a - 0.5p1 - 0.5p2, p1 - p0, p3 - p2,
        ]
    end
    @test ForwardDiff.jacobian(constraints, q) ≈ ForwardDiff.jacobian(analytic, q)
    bang = let model = model, rebuild = rebuild, state = st
        x -> optimization_constraints!(similar(x, length(model.lcons)), model, rebuild(x), state)
    end
    @test ForwardDiff.jacobian(bang, q) ≈ ForwardDiff.jacobian(analytic, q)
    @test_throws ArgumentError optimization_constraints(model, ps, merge(st, (; timestops = sort(vcat(st.timestops, 0.4)))))
    @test_throws ArgumentError CorleoneDynamicOptProblem(layer, :u₁; ps)
    @test_throws ArgumentError CorleoneDynamicOptProblem(layer, traj -> traj.u; ps, st)
    @test_throws ArgumentError CorleoneDynamicOptProblem(layer; ps, st)
    @test_throws ArgumentError CorleoneDynamicOptProblem(layer, :(u₁(1.0) <= 2.0); ps, st)
    @test_throws ArgumentError CorleoneDynamicOptProblem(layer, :u₁, :(u₁(1.0) == 0.0); ps, st, lcons = [0.0], ucons = [0.0])
    @test_throws DomainError CorleoneDynamicOptProblem(layer, :(u₁(4.0)); ps, st)
    @test_throws ArgumentError CorleoneDynamicOptProblem(layer, :(u₁(NaN)); ps, st)
    @test_throws ArgumentError CorleoneDynamicOptProblem(layer, :(u₁(1.0, 2.0)); ps, st)
    @test_throws ArgumentError CorleoneDynamicOptProblem(layer, :u₁, traj -> 1.0; ps, st)
    @test_throws DimensionMismatch CorleoneDynamicOptProblem(layer, :u₁, inplace!, scalar!; ps, st, lcons = [[0.0, 0.0], 0.0], ucons = [[1.0], 1.0])
    @test_throws ArgumentError CorleoneDynamicOptProblem(layer, :u₁, scalar!; ps, st, lcons = [2.0], ucons = [1.0])
    @test_throws ArgumentError CorleoneDynamicOptProblem(layer, :u₁, scalar!; ps, st, lcons = [NaN], ucons = [1.0])
    @test_throws DimensionMismatch CorleoneDynamicOptProblem(layer, :u₁, traj -> [1.0, 2.0]; ps, st, lcons = [0.0], ucons = [1.0])
    @test_throws ArgumentError CorleoneDynamicOptProblem(layer, :(u₁(3 / 2)); ps, st)
    fractional = CorleoneDynamicOptProblem(layer, :(u₁($(1 // 3))); ps, st)
    @test optimization_objective(fractional) ≈ 1 + 0.5 / 3
    series = CorleoneDynamicOptProblem(layer, :(sum(u₁)); ps, st)
    series_traj, _ = layer((layer.problem, series.evaluation_grid), ps, st)
    @test optimization_objective(series) ≈ sum(first, series_traj.u)
    @test calls[] == 1
end

@testset "Named signals and evaluator contexts" begin
    sys = SymbolCache([:x, :u], [:p_1], :t; defaults = Dict(:x => 1.0, :u => 2.0, :p_1 => 0.5))
    f = ODEFunction((u, p, t) -> [p[1], -p[1]]; sys)
    layer = ShootingLayer(ODEProblem(f, [1.0, 2.0], (0.0, 25.0), [0.5]), Tsit5())
    @test Corleone.expression_symbol_indices(layer)[:p_1] == (:parameters, 1)
    @test Corleone.expression_symbol_indices(layer)[:u] == (:states, 2)
    for evaluator in (Val(:Eval), Val(:RuntimeGeneratedFunctions))
        model = CorleoneDynamicOptProblem(
            layer, :(facade_helper(x(3.0))),
            :(p_1(5.0) * x(15.0)), :(sin(x(3.0)) - u(25.0));
            lcons = [0.0, 0.0], ucons = [10.0, 20.0], eval_module = @__MODULE__, evaluator
        )
        @test optimization_objective(model) ≈ 6.0
        @test optimization_constraints(model) ≈ [4.25, sin(2.5) + 10.5]
        @test model.evaluation_grid == [0.0, 3.0, 5.0, 15.0, 25.0]
        @test model.timepoints[:x] == [3.0, 15.0]
        conditional = CorleoneDynamicOptProblem(layer, :(ifelse(x(3.0) > 0, x(3.0), 0.0)); evaluator)
        @test optimization_objective(conditional) ≈ 2.5
    end
    @test default_evaluator() === Val(:RuntimeGeneratedFunctions)
end

@testset "Quadrature defaults and empty constraints" begin
    layer = ShootingLayer(ODEProblem((u, p, t) -> [p[1], u[1]^2], [1.0, 0.0], (0.0, 1.0), [0.5]), Tsit5(); quadratures = [2])
    d = CorleoneDynamicOptProblem(layer; rng = MersenneTwister(2))
    traj, _ = layer(nothing, d.ps, d.st)
    @test optimization_objective(d) ≈ last(traj.u)[2]
    @test isempty(optimization_constraints(d))
    @test optimization_constraints!(Float64[], d) == Float64[]
    terminal = CorleoneDynamicOptProblem(layer, :u₂, :u₁; ps = d.ps, st = d.st, lcons = [0.0], ucons = [2.0])
    @test only(optimization_constraints(terminal)) ≈ last(traj.u)[1]
end
