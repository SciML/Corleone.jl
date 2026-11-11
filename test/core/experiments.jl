using Test, Corleone, LuxCore, OrdinaryDiffEqTsit5, StableRNGs

# Exercise the observation API using only core Corleone, without CorleoneInverse.
problem = ODEProblem((u, p, t) -> [p[1]], [1.0], (0.0, 2.0), [2.0])
layer = ShootingLayer(problem, Tsit5())
ps, st = LuxCore.setup(StableRNG(1), layer)

@testset "Observation grids, selectors, and weighted residuals" begin
    experiment = Experiment(
        :u₁ => ([0.0, 1.0], [1.5, 3.5], 0.5),
        ((u, p, t) -> u[1] + p[1]) => ([0.5, 2.0], [4.0, 8.0], [1.0, 2.0])
    )
    @test experiment.t == [0.0, 0.5, 1.0, 2.0]
    @test n_observations(experiment) == 4
    state = experiment_state(layer, experiment, st)
    @test state.timestops === st.timestops
    @test state.saveat == experiment.t
    traj, _ = layer(experiment, ps, state)
    @test traj.t == experiment.t
    @test residual(experiment, traj) ≈ [1.0, 1.0, 0.0, 0.5]
    @test residual(first(experiment.observations), traj) ≈ [1.0, 1.0]
    @test residual((experiment, experiment), (traj, traj)) ≈ repeat(residual(experiment, traj), 2)
    @test residual((a = experiment,), (a = traj,)) == residual(experiment, traj)
    @test n_observations((experiment, experiment)) == 8
    @test n_observations((a = experiment,)) == 4
end

@testset "Observation times and configured solver samples coexist" begin
    prob = ODEProblem(problem.f, problem.u0, (0.0f0, 2.0f0), problem.p; saveat = [0.25, 0.5])
    shooting = ShootingLayer(prob, Tsit5())
    ps, st = LuxCore.setup(StableRNG(1), shooting)
    data = Experiment(1 => ([0.1, 0.5, 1.5], [1.2, 2.0, 4.0]))
    state = experiment_state(shooting, data, st)
    traj, _ = shooting(data, ps, state)
    @test state.timestops == [0.0, 2.0]
    @test traj.t == [0.0, 0.1, 0.25, 0.5, 1.5, 2.0]
    @test residual(data, traj) ≈ zeros(3) atol = 1.0e-12
    additional = Experiment(1 => ([0.75], [2.5]))
    @test experiment_state(shooting, additional, state).saveat == [0.1, 0.25, 0.5, 0.75, 1.5]

    stepped = remake(problem; saveat = 0.5)
    shooting = ShootingLayer(stepped, Tsit5(), Constant(1; default = [2.0, 2.0]) => [0.75]; controls = [1])
    ps, st = LuxCore.setup(StableRNG(1), shooting)
    state = experiment_state(shooting, data, st)
    @test state.timestops == [0.0, 0.75, 2.0]
    @test state.saveat == [0.1, 0.5, 1.25, 1.5, 1.75]
    traj, _ = shooting(data, ps, state)
    @test traj.t == [0.0, 0.1, 0.5, 0.75, 1.25, 1.5, 1.75, 2.0]
    @test residual(data, traj) ≈ zeros(3) atol = 1.0e-12
end

@testset "Experiment initial states and unweighted observations" begin
    experiment = Experiment(1 => ([0.0, 2.0], [3.0, 7.0]); u0 = [3.0])
    traj, _ = layer(experiment, ps, experiment_state(layer, experiment, st))
    @test first(traj.u) == [3.0]
    @test residual(experiment, traj) ≈ zeros(2) atol = 1.0e-12
    @test problem.u0 == [1.0]
end

@testset "Observation and experiment validation" begin
    @test Observation(1 => ([0.0], [1.0], 0.5)).σ == 0.5
    @test_throws ArgumentError Observation(1 => ())
    @test_throws ArgumentError Observation(1 => ([0.0],))
    @test_throws ArgumentError Observation(1 => ([0.0], [1.0], 0.5, :extra))
    @test_throws ArgumentError Observation(1 => ([0.0], [1.0], -0.5))
    @test_throws ArgumentError Observation(1 => ([0.0], [1.0], NaN))
    @test_throws ArgumentError Observation(1 => ([0.0], [1.0], [0.0]))
    @test_throws ArgumentError Experiment()
    @test_throws DimensionMismatch Observation(1 => ([0.0, 1.0], [1.0]))
    @test_throws ArgumentError Observation(1 => ([1.0, 0.0], [1.0, 2.0]))
    @test_throws ArgumentError Observation(1 => ([0.0, 0.0], [1.0, 2.0]))
    @test_throws ArgumentError Observation(1 => ([0.0], [1.0], 0.0))
    @test_throws DimensionMismatch Observation(1 => ([0.0], [1.0], [0.1, 0.2]))
    @test_throws ArgumentError experiment_state(layer, Experiment(1 => ([3.0], [1.0])), st)
    @test_throws DimensionMismatch experiment_state(layer, Experiment(1 => ([0.0], [1.0]); u0 = [1.0, 2.0]), st)
end
