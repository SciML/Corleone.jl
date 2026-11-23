using Test, Corleone, DataFrames, LuxCore, OrdinaryDiffEqTsit5, Random

measurements = DataFrame(
    observableId = ["x", "x", "x", "x", "x", "offset"],
    experimentId = ["a", "b", "a", "a", "a", "a"],
    time = [1.0, 0.5, 0.0, 1.0, 1.0, 0.5],
    measurement = [3.5, 4.0, 1.5, 4.0, 4.5, 4.0],
    noiseParameters = Any[0.5, "2.0", missing, "", 1.5, 1.0],
    replicateId = [1, 1, 1, 2, 3, 1],
)
selectors = Dict("x" => 1, "offset" => ((u, p, t) -> u[1] + p[1]))
problem = ODEProblem((u, p, t) -> [p[1]], [1.0], (0.0, 2.0), [2.0])
layer = ShootingLayer(problem, Tsit5())
ps, st = LuxCore.setup(Xoshiro(1), layer)

@testset "Import and independent replicate residuals" begin
    original = copy(measurements)
    data = experiments_from_dataframe(
        measurements; observables = selectors,
        initial_states = Dict(:a => [1.0], :b => [3.0])
    )
    @test keys(data) == (:a, :b)
    @test isequal(measurements, original)
    @test n_observations(data) == 6
    @test data.a.t == [0.0, 0.5, 1.0]
    @test all(o -> issorted(o.t) && allunique(o.t), data.a.observations)
    traj, _ = layer(data.a, ps, experiment_state(layer, data.a, st))
    @test residual(data.a, traj) ≈ [0.5, 1.0, 1.0, 1.0, 0.0]
    other, _ = layer(data.b, ps, experiment_state(layer, data.b, st))
    @test residual(data.b, other) ≈ [0.0] atol = 1.0e-12
    @test Experiment(view(measurements, [2], :); observables = selectors, u0 = [3.0]).u0 == [3.0]
    batch = Experiments(layer, data)
    bp, bs = LuxCore.setup(Xoshiro(2), batch)
    trajectories, _ = layer(batch, bp, bs)
    @test length(trajectories) == 2
    @test first(trajectories[2].u) == [3.0]
    raw = select(measurements[[1, 3], :], Not(:noiseParameters))
    raw.observableId .= "u₁"
    one = Experiment(raw)
    @test only(one.observations).selector == :u₁
    @test isnothing(only(one.observations).σ)
    @test Experiment(raw; observables = (u₁ = 1,)).observations[1].selector == 1
end

@testset "Export and round trip" begin
    traj, _ = layer(nothing, ps, merge(st, (; saveat = [0.0, 0.5, 1.0, 2.0])))
    frame = DataFrame(traj; experiment_id = "a")
    @test names(frame) == ["observableId", "experimentId", "time", "simulation", "sampleIndex", "resetSide"]
    @test frame.observableId == fill("u₁", length(traj.t))
    @test DataFrame([traj]; experiment_ids = ["a"]) == frame
    @test frame.time == traj.t
    @test frame.simulation ≈ first.(traj.u)
    @test frame.sampleIndex == collect(eachindex(traj.t))
    @test all(==("none"), frame.resetSide)
    callback = trajectories_dataframe(traj; observables = selectors)
    @test callback.simulation[callback.observableId .== "x"] ≈ first.(traj.u)
    @test callback.simulation[callback.observableId .== "offset"] ≈ first.(traj.u) .+ 2
    @test trajectories_dataframe(traj; observables = [:x => :u₁]).observableId == fill("x", length(traj.t))
    many = trajectories_dataframe((a = traj, b = traj))
    @test unique(many.experimentId) == ["a", "b"]
    @test unique(trajectories_dataframe([traj, traj]).experimentId) == ["experiment1", "experiment2"]
    @test unique(trajectories_dataframe((traj, traj); experiment_ids = [:c, :d]).experimentId) == ["c", "d"]
    data = experiments_from_dataframe(rename(many, :simulation => :measurement))
    @test n_observations(data) == 2length(traj.t)
    @test residual(data.a, traj) ≈ zeros(length(traj.t))
    expression_data = experiments_from_dataframe(
        rename(frame, :simulation => :measurement);
        observables = Dict("u₁" => :(u₁))
    )
    @test only(expression_data.a.observations).selector == :(u₁)
    @test trajectories_dataframe(traj; observables = (x = 1,)).simulation == frame.simulation
    @test_throws ArgumentError trajectories_dataframe(traj; observables = Dict("x" => :unknown))
end

@testset "Both reset samples survive export" begin
    reset_layer = ShootingLayer(
        problem, Tsit5(),
        Constant(1; kind = Val(:variable), default = 10.0) => [1.0]
    )
    rp, rs = LuxCore.setup(Xoshiro(3), reset_layer)
    traj, _ = reset_layer(nothing, rp, rs)
    frame = trajectories_dataframe(traj)
    k = only(traj.reset_indices)
    @test frame.time[k - 1] == frame.time[k] == 1.0
    @test frame.simulation[k - 1] ≈ 3.0
    @test frame.simulation[k] ≈ 10.0
    @test frame.resetSide[(k - 1):k] == ["incoming", "outgoing"]
    @test frame.sampleIndex[(k - 1):k] == [k - 1, k]
end

@testset "Validation" begin
    @test Base.get_extension(Corleone, :CorleoneDataFramesExtension) !== nothing
    @test_throws ArgumentError experiments_from_dataframe(DataFrame())
    @test_throws ArgumentError experiments_from_dataframe(measurements[1:0, :])
    @test_throws ArgumentError Experiment(measurements; observables = selectors)
    @test_throws ArgumentError experiments_from_dataframe(measurements; observables = Dict())
    @test_throws ArgumentError experiments_from_dataframe(measurements; initial_states = Dict())
    for (column, bad) in (
            (:time, Inf), (:measurement, missing), (:observableId, ""),
            (:experimentId, missing), (:noiseParameters, "sigma"), (:noiseParameters, -1.0),
        )
        frame = DataFrame([name => Any[measurements[1, name]] for name in names(measurements)])
        frame[1, column] = bad
        @test_throws ArgumentError experiments_from_dataframe(frame)
    end
    for column in (:observableParameters, :preequilibrationConditionId, :simulationConditionId)
        frame = copy(measurements)
        frame[!, column] = fill("unsupported", nrow(frame))
        @test_throws ArgumentError experiments_from_dataframe(frame)
    end
    traj, _ = layer(nothing, ps, st)
    @test_throws ArgumentError trajectories_dataframe(Trajectory[])
    @test_throws ArgumentError trajectories_dataframe([1])
    @test_throws DimensionMismatch trajectories_dataframe([traj]; experiment_ids = ["a", "b"])
    @test_throws ArgumentError trajectories_dataframe([traj, traj]; experiment_ids = ["a", "a"])
    @test_throws ArgumentError trajectories_dataframe(traj; observables = [:a => 1, "a" => 1])
    @test_throws ArgumentError trajectories_dataframe(traj; observables = Dict("x" => :(u₁ + 1)))
end
