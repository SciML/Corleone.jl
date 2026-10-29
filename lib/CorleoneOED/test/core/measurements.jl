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

@testset "Mixed observation rows and direct parameter dependence" begin
    problem = ODEProblem((u, p, t) -> p[1] .* u, [1.0], (0.0, 1.0), [-2.0]; abstol = 1.0e-11, reltol = 1.0e-11)
    layer = OEDLayer(
        problem, Tsit5(); measurements = (
            ContinuousMeasurement(:vector_density, [0.5], (u, p, t) -> [u[1], p[1] * u[1]]),
            ContinuousMeasurement(:scalar_density, [], (u, p, t) -> p[1]),
            DiscreteMeasurement(:events, [0.0, 0.5, 1.0], (u, p, t) -> [u[1] + p[1], 2p[1]]),
        )
    )
    ps, st = LuxCore.setup(Xoshiro(1), layer)
    parameters = ps.parameter.parameters
    parameters.vector_density.parameter.ps .= [0.2, 0.8]
    parameters.scalar_density.parameter.ps .= 0.4
    parameters.events.parameter.ps .= [0.3, 0.6, 0.9]
    trajectory, _ = layer(nothing, ps, st)
    # Integral of exp(-4t) * (t^2 + (1 - 2t)^2).
    primitive(t) = exp(-4t) * (-5t^2 / 4 + 3t / 8 - 5 / 32)
    continuous = [primitive(0.5) - primitive(0.0), primitive(1.0) - primitive(0.5)]
    discrete = [(1 + t * exp(-2t))^2 + 4 for t in (0.0, 0.5, 1.0)]
    reference = dot([0.2, 0.8], continuous) + 0.4 + dot([0.3, 0.6, 0.9], discrete)
    @test only(fisher_information(layer, trajectory)) ≈ reference rtol = 1.0e-8
    @test sampling_sums(layer, trajectory) ≈ [0.5, 0.4, 1.8]
    @test sampling_sums(layer, nothing, ps, st) ≈ sampling_sums(layer, trajectory)
    @test first(observed_equations(layer, trajectory)) == [1.0, -2.0, -2.0, -1.0, -4.0]

    gains, _ = local_information_gain(layer, nothing, ps, st)
    global_gains, _ = global_information_gain(layer, nothing, ps, st)
    for (t, local_rows, global_rows) in zip(trajectory.t, gains, global_gains)
        J = [t * exp(-2t), (1 - 2t) * exp(-2t), 1.0, 1 + t * exp(-2t), 2.0]
        @test only.(local_rows) ≈ J .^ 2 rtol = 1.0e-8 atol = 1.0e-10
        @test only.(global_rows) ≈ (J ./ reference) .^ 2 rtol = 1.0e-8 atol = 1.0e-10
    end
    q, reconstruct = pack(default_vectorizer(), ps)
    loss(q) = only(first(fisher_information(layer, nothing, reconstruct(q), st)))
    gradient = ForwardDiff.gradient(loss, q)
    expected = zero(ComponentArray(ps))
    expected.parameter.parameters.vector_density.parameter.ps .= continuous
    expected.parameter.parameters.scalar_density.parameter.ps .= 1.0
    expected.parameter.parameters.events.parameter.ps .= discrete
    @test gradient ≈ collect(expected) rtol = 1.0e-8
    gain_loss(q) = sum(first(global_information_gain(layer, nothing, reconstruct(q), st))) do rows
        sum(only, rows)
    end
    @test ForwardDiff.gradient(gain_loss, q) ≈ -2gain_loss(q) / reference * collect(expected) rtol = 1.0e-8
end

@testset "Discrete observations select the outgoing reset state" begin
    problem = ODEProblem((u, p, t) -> p[1] .* u, [1.0], (0.0, 1.0), [-2.0]; abstol = 1.0e-11, reltol = 1.0e-11)
    reset = Tunable(1; kind = Val(:variable), default = 3.0) => [0.5]
    layer = OEDLayer(
        problem, Tsit5(), reset; measurements = (
            DiscreteMeasurement(:event, [0.5], (u, p, t) -> p[1] * u[1]),
        )
    )
    ps, st = LuxCore.setup(Xoshiro(1), layer)
    trajectory, _ = layer(nothing, ps, st)
    indices = findall(==(0.5), trajectory.t)
    @test length(indices) == 2
    @test first(trajectory.u[first(indices)]) ≈ exp(-1)
    @test first(trajectory.u[last(indices)]) == 3.0
    @test only(fisher_information(layer, trajectory)) ≈ 9.0
end
