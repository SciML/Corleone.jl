using CorleoneOED, LuxCore, Random, ComponentArrays, OrdinaryDiffEqTsit5
using Test, ForwardDiff, LinearAlgebra, OrdinaryDiffEqBDF
@testset "Measurement augmentation" begin
    problem = ODEProblem((u, p, t) -> p[1] .* u, [1.0], (0.0, 1.0), [-2.0]; abstol = 1.0e-10, reltol = 1.0e-10)
    model = OEDLayer(
        problem, Tsit5(); params = [1], measurements = (
            ContinuousMeasurement(:wc, [0.0, 0.5], (u, p, t) -> [u[1], 2u[1]]),
            DiscreteMeasurement(:wd, [0.0, 0.5, 1.0], (u, p, t) -> [u[1], 2u[1]]),
        )
    )
    ps, st = LuxCore.setup(Xoshiro(1), model)
    traj, _ = model(nothing, ps, st)
    reference = 5 * ((2 - 26exp(-4)) / 64 + 0.25exp(-2) + exp(-4))
    @test only(fisher_information(model, traj)) ≈ reference rtol = 1.0e-7
    @test sampling_sums(model, nothing, ps, st) ≈ [1.0, 3.0]
    @test continuous_sampling_sums(model, traj) ≈ [1.0]
    @test discrete_sampling_sums(model, traj) ≈ [3.0]
    q, unpack = pack(default_vectorizer(), ps)
    loss(q) = only(first(fisher_information(model, nothing, unpack(q), st)))
    gradient = ForwardDiff.gradient(loss, q)
    @test all(isfinite, gradient)
    @test dot(gradient, q) ≈ loss(q) rtol = 1.0e-7
    @test update_fim(model, [], st) === st
    @test update_fim(model, [(; ps, st)], st).F_init ≈ fisher_information(model, traj)
    direct = OEDLayer(problem, Tsit5(); params = [1], measurements = (DiscreteMeasurement(:w, [0.0, 1.0], (u, p, t) -> p[1]),))
    dps, dst = LuxCore.setup(Xoshiro(1), direct)
    @test only(first(fisher_information(direct, nothing, dps, dst))) ≈ 2.0
    @test_throws ArgumentError OEDLayer(problem, Tsit5(); measurements = (DiscreteMeasurement(:w, [0.5, 0.0], (u, p, t) -> u[1]),))
    @test_throws ArgumentError OEDLayer(problem, Tsit5(), Constant(1; kind = Val(:variable), default = 1.0) => [0.5]; measurements = ())
    @test_throws ArgumentError OEDLayer(problem, Tsit5(), Constant(1; kind = Val(:variable), default = 1.0) => [1.0]; measurements = ())
    # A nonidentity constant mass matrix scales both state and sensitivity dynamics.
    massprob = ODEProblem(ODEFunction((u, p, t) -> p[1] .* u; mass_matrix = reshape([2.0], 1, 1)), [1.0], (0.0, 1.0), [-2.0]; abstol = 1.0e-10, reltol = 1.0e-10)
    masslayer = OEDLayer(massprob, FBDF(); params = [1], measurements = (DiscreteMeasurement(:w, [1.0], (u, p, t) -> u[1]),))
    mps, mst = LuxCore.setup(Xoshiro(1), masslayer)
    @test only(first(fisher_information(masslayer, nothing, mps, mst))) ≈ 0.25exp(-2) rtol = 1.0e-6
end
@testset "DAE augmentation" begin
    residual(du, u, p, t) = [p[1] * u[1] - du[1], u[2] - u[1]]
    problem = DAEProblem(residual, [-2.0, -2.0], [1.0, 1.0], (0.0, 1.0), [-2.0]; differential_vars = [true, false], abstol = 1.0e-9, reltol = 1.0e-9)
    layer = OEDLayer(problem, DFBDF(); params = [1], measurements = (DiscreteMeasurement(:w, [1.0], (u, p, t) -> u[2]),))
    ps, st = LuxCore.setup(Xoshiro(1), layer)
    @test layer.augmented_prob.differential_vars == [true, false, true, false, true]
    @test only(first(fisher_information(layer, nothing, ps, st))) ≈ exp(-4) rtol = 1.0e-5
end
@testset "Reduced parameter directions" begin
    problem = ODEProblem((u, p, t) -> [sum(p) * u[1]], [1.0], (0.0, 1.0), [-1.0, -0.5, -0.5]; abstol = 1.0e-10, reltol = 1.0e-10)
    layer = OEDLayer(
        problem, Tsit5(); params = [1, 2, 3], svd = true, ns = 1,
        measurements = (DiscreteMeasurement(:w, [1.0], (u, p, t) -> u[1]),)
    )
    ps, st = LuxCore.setup(Xoshiro(1), layer)
    @test size(layer.basis) == (3, 1)
    @test size(first(fisher_information(layer, nothing, ps, st))) == (1, 1)
    @test only(first(fisher_information(layer, nothing, ps, st))) ≈ 3exp(-4) rtol = 1.0e-6
end

@testset "Augmented shooting continuity" begin
    problem = ODEProblem((u, p, t) -> p[1] .* u, [1.0], (0.0, 1.0), [-2.0]; abstol = 1.0e-10, reltol = 1.0e-10)
    reset = Tunable(1; kind = Val(:variable), default = 1.0) => [0.5]
    layer = OEDLayer(
        problem, Tsit5(), reset; params = [1], measurements = (
            ContinuousMeasurement(:w, [0.0, 0.5], (u, p, t) -> u[1]),
        )
    )
    ps, st = LuxCore.setup(Xoshiro(1), layer)
    forward_initialization(Xoshiro(1), layer.shooting, ps, st)
    traj, _ = layer(nothing, ps, st)
    @test length(shooting_constraints(traj)) == get_number_of_shooting_constraints(layer, ps, st) == 3
    @test norm(shooting_constraints(traj)) < 1.0e-8
    @test only(fisher_information(layer, traj)) ≈ (2 - 26exp(-4)) / 64 rtol = 1.0e-7
end

@testset "Wrap an annotated shooting layer" begin
    problem = ODEProblem((u, p, t) -> p[1] .* u, [1.0], (0.0, 1.0), [-2.0])
    parameter = Tunable(1; default = -2.0, lowerbound = -3.0, upperbound = -1.0) => [0.5]
    reset = Tunable(1; kind = Val(:variable), default = 0.5, lowerbound = 0.0, upperbound = 2.0) => [0.5]
    original = ShootingLayer(problem, Tsit5(), parameter, reset; controls = [1])
    layer = OEDLayer(original; params = [1], measurements = (DiscreteMeasurement(:w, [1.0], (u, p, t) -> u[1]),))
    ps, st = LuxCore.setup(Xoshiro(1), layer)
    lb, ub = get_bounds(layer, ps, st) .|> ComponentArray
    @test minimum(lb.parameter) == -3.0
    @test get_bounds(layer.shooting.sys, :u₁) == (0.0, 2.0)
    @test :p₁ in control_symbols(layer.shooting.sys)
    @test get_number_of_shooting_constraints(layer, ps, st) == 3
    forward_initialization(Xoshiro(1), layer.shooting, ps, st)
    traj, _ = layer(nothing, ps, st)
    @test norm(shooting_constraints(traj)) < 1.0e-7
end
