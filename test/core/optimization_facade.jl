using Test, Corleone, LuxCore, OrdinaryDiffEqTsit5, ComponentArrays, Random, ForwardDiff

@testset "Prepared trajectory objectives and constraints" begin
    calls = Ref(0)
    initialize = (rng, n::Int) -> (calls[] += 1; fill(0.5, n))
    prob = ODEProblem(
        (u, p, t) -> [p[1]], [1.0], (0.0, 3.0), [0.5];
        saveat = [0.25, 0.75, 1.25, 1.75, 2.25, 2.75], abstol = 1.0e-11, reltol = 1.0e-11
    )
    layer = ShootingLayer(
        prob, Tsit5(),
        Tunable(1; kind = Val(:variable), default = [3.0, 4.0]) => [1.0, 2.0],
        Tunable(1; kind = Val(:parameter), default = initialize, lowerbound = 0.0, upperbound = 2.0) => [1.5]
    )
    ps, st = LuxCore.setup(MersenneTwister(1), layer)
    @test calls[] == 1
    ps, st = add_parameter_segments!(layer, ps, st)
    d = CorleoneDynamicOptProblem(
        layer, :u₁,
        :u₁ => (; t = [1.0, 2.0], bounds = (0.0, 10.0)),
        :u₁ => (; t = 1.0, bounds = (0.0, 10.0), side = :pre),
        :u₁ => (; t = 1.0, bounds = ([0.0, 0.0], [10.0, 10.0]), side = :both),
        :p₁ => (; t = [1.5, 3.0], bounds = (0.0, 2.0)),
        (traj -> last(traj.u)[1]^2) => (; bounds = (0.0, 100.0)); ps, st
    )
    @test calls[] == 1
    @test d.ps === ps && d.st === st && d.layer === layer
    traj, _ = layer(nothing, ps, st)
    expected = vcat([3.0, 4.0, 1.5, 1.5, 3.0, 0.5, 0.5, 20.25], shooting_constraints(traj))
    @test optimization_objective(d) ≈ 4.5
    @test optimization_constraints(d) ≈ expected
    @test length(d.lcons) == length(d.ucons) == length(expected)
    @test d.lcons == zeros(length(expected))
    @test d.ucons == vcat(fill(10.0, 5), fill(2.0, 2), 100.0, zeros(4))
    buffer = fill(NaN, length(expected))
    @test optimization_constraints!(buffer, d) === buffer
    @test buffer ≈ expected
    wrong = fill(NaN, length(buffer) - 1)
    @test_throws DimensionMismatch optimization_constraints!(wrong, d)
    @test all(isnan, wrong)
    # The right control value is selected at switches, independently of state sample layout.
    changed = deepcopy(ps)
    changed.parameter.parameters.p₁.parameter.ps .= [0.1, 0.2, 0.7, 0.8]
    @test optimization_constraints(d, changed, st)[6:7] ≈ [0.7, 0.8]
    @test calls[] == 1
    q = ComponentArray(ps);axes = getaxes(q)
    objective, constraints = let model = d, parameter_axes = axes, state = st
        (
            x -> optimization_objective(model, ComponentArray(x, parameter_axes), state),
            x -> optimization_constraints(model, ComponentArray(x, parameter_axes), state),
        )
    end
    @test (@inferred objective(q)) ≈ 4.5
    jac = ForwardDiff.jacobian(constraints, q)
    function analytic_constraints(x)
        tree = ComponentArray(x, axes)
        a, b = tree.initialcondition.initialconditions.u₁.parameter.ps
        p0, p1, p2, p3 = tree.parameter.parameters.p₁.parameter.ps
        return [
            a, b, 1 + p0, 1 + p0, a, p2, p3, (b + p3)^2,
            a - 1 - p0, b - a - 0.5p1 - 0.5p2, p1 - p0, p3 - p2,
        ]
    end
    @test jac ≈ ForwardDiff.jacobian(analytic_constraints, q)
    function inplace_constraints(x)
        res = similar(x, length(d.lcons))
        optimization_constraints!(res, d, ComponentArray(x, axes), st)
    end
    @test ForwardDiff.jacobian(inplace_constraints, q) ≈ jac
    @test any(!iszero, jac[:, 1])
    @test_throws ArgumentError CorleoneDynamicOptProblem(layer, :u₁, :u₁ => (; t = 0.3, bounds = (0, 1)); ps, st)
    @test_throws ArgumentError CorleoneDynamicOptProblem(layer, :u₁, :u₁ => (; t = 1, bounds = (0, 1), side = :invalid); ps, st)
    @test_throws ArgumentError CorleoneDynamicOptProblem(layer, :u₁, :u₁ => (; t = 1, bounds = (2, 1)); ps, st)
    @test_throws ArgumentError CorleoneDynamicOptProblem(layer, :u₁; ps)
    @test_throws ArgumentError CorleoneDynamicOptProblem(layer, traj -> traj.u; ps, st)
    @test_throws ArgumentError CorleoneDynamicOptProblem(layer; ps, st)
    @test_throws DimensionMismatch CorleoneDynamicOptProblem(layer, :u₁, :u₁ => (; t = [1, 2], bounds = ([0.0], [1.0])); ps, st)
    @test_throws ArgumentError CorleoneDynamicOptProblem(layer, :u₁, :u₁ => (; t = [1, 1], bounds = (0, 1)); ps, st)
    allsamples = CorleoneDynamicOptProblem(layer, :u₁, :u₁ => (; t = :all, bounds = (-Inf, Inf)); ps, st)
    @test length(first(allsamples.constraints)) == length(unique(traj.t))
    callback = CorleoneDynamicOptProblem(
        layer, traj -> last(traj.u)[1],
        (traj -> first.(traj.u)) => (; t = [1, 2], bounds = (0, 10)); ps, st
    )
    @test optimization_constraints(callback)[1:2] ≈ [3, 4]
    changed_st = merge(st, (; timestops = sort(unique(vcat(st.timestops, 0.4)))))
    @test_throws ArgumentError optimization_constraints(d, ps, changed_st)
    dynamic = CorleoneDynamicOptProblem(
        layer, :u₁,
        (traj -> last(traj.u)[1] > 4 ? [1.0] : [1.0, 2.0]) => (; bounds = (0, 3)); ps, st
    )
    shorter = deepcopy(ps)
    shorter.initialcondition.initialconditions.u₁.parameter.ps[end] = 1.0
    untouched = fill(NaN, length(dynamic.lcons))
    @test_throws DimensionMismatch optimization_constraints(dynamic, shorter, st)
    @test_throws DimensionMismatch optimization_constraints!(untouched, dynamic, shorter, st)
    @test all(isnan, untouched)
end

@testset "Quadrature defaults and empty constraint contracts" begin
    prob = ODEProblem((u, p, t) -> [p[1], u[1]^2], [1.0, 0.0], (0.0, 1.0), [0.5]; abstol = 1.0e-11, reltol = 1.0e-11)
    layer = ShootingLayer(prob, Tsit5(); quadratures = [2])
    d = CorleoneDynamicOptProblem(layer; rng = MersenneTwister(2))
    traj, _ = layer(nothing, d.ps, d.st)
    @test optimization_objective(d) ≈ last(traj.u)[2]
    @test isempty(optimization_constraints(d))
    @test optimization_constraints!(Float64[], d) == Float64[]
    terminal = CorleoneDynamicOptProblem(layer, :u₂, :u₁ => (; bounds = (0.0, 2.0)); ps = d.ps, st = d.st)
    @test optimization_constraints(terminal) ≈ [1.5]
end
