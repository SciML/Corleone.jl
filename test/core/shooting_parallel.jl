using Corleone, LuxCore, OrdinaryDiffEqTsit5, Random, Test

with_ensemble(layer, alg) = ShootingLayer(layer.name, layer.problem, layer.alg, alg,
    layer.initialcondition, layer.parameter)

@testset "Independent shooting blocks" begin
    prob = ODEProblem((u, p, t) -> [p[1] * u[1], u[1] - u[2]],
        [1.0, 2.0], (0.0, 4.0), [0.2]; abstol = 1e-10, reltol = 1e-10,
        saveat = [0.25, 1.25, 2.25, 3.25])
    layer = ShootingLayer(prob, Tsit5(),
        Tunable(1; kind = Val(:variable)) => [1.0, 2.0, 3.0, 4.0],
        Constant(2; kind = Val(:variable)) => [2.0, 4.0],
        Tunable(1; kind = Val(:parameter)) => [0.5, 1.5, 2.5, 3.5])
    ps, st = LuxCore.setup(MersenneTwister(1), layer)
    stops = Corleone.get_timestops(layer, ps, st)
    @test Corleone.shooting_blocks(layer, ps, st, stops) == [1:5, 5:9]
    local_st = Corleone.private_shooting_state(layer, st)
    @test local_st.timestops === st.timestops
    @test local_st.initialcondition.active !== st.initialcondition.active
    @test local_st.initialcondition.initialconditions.u₂.parameters ===
        st.initialcondition.initialconditions.u₂.parameters
    serial, serial_st = layer(nothing, ps, st)
    threaded = with_ensemble(layer, EnsembleThreads())
    for _ in 1:3
        before = copy(st.initialcondition.active)
        actual, actual_st = threaded(nothing, ps, st)
        @test actual.t == serial.t
        @test actual.u ≈ serial.u
        @test actual.residuals ≈ serial.residuals
        @test actual.reset_indices == serial.reset_indices
        @test actual_st == serial_st
        @test actual_st.timestops === st.timestops
        @test actual.ps === ps
        @test st.initialcondition.active == before
        @test actual_st.initialcondition.active !== st.initialcondition.active
    end
    @test 0.25 in serial.t # Retain actual solver samples when joining blocks.
    @test last(serial.t) == 4.0 # A final-time reset never starts an empty block.
    @test length(serial.residuals) == length(stops) - 1

    # Reuse the constructed layer after moving a fixed variable's grid in state.
    fixed = st.initialcondition.initialconditions.u₂
    lookup = merge(fixed.states.lookup, (; collection = Dict(1.5 => 1, 3.0 => 2)))
    fixed = merge(fixed, (; states = merge(fixed.states, (; lookup))))
    children = merge(st.initialcondition.initialconditions, (; u₂ = fixed))
    changed_st = merge(st, (; initialcondition = merge(st.initialcondition,
        (; initialconditions = children))))
    changed_stops = Corleone.get_timestops(threaded, ps, changed_st)
    changed_st = merge(changed_st, (; timestops = changed_stops))
    @test Corleone.shooting_blocks(threaded, ps, changed_st, changed_stops) == [1:7, 7:9]
    expected, _ = layer(nothing, ps, changed_st)
    actual, _ = threaded(nothing, ps, changed_st)
    @test actual.u ≈ expected.u
    @test actual.t == expected.t
    @test actual.residuals ≈ expected.residuals
    _, updated = threaded(nothing, ps, changed_st)
    repeated, _ = threaded(nothing, ps, updated)
    @test repeated.u ≈ expected.u
    @test updated.timestops === changed_stops
end

@testset "Partial resets remain sequential" begin
    prob = ODEProblem((u, p, t) -> [p[1], u[1]], [1.0, 2.0], (0.0, 3.0), [0.2])
    layer = ShootingLayer(prob, Tsit5(),
        Tunable(1; kind = Val(:variable)) => [1.0, 2.0]; ensemblealg = EnsembleThreads())
    ps, st = LuxCore.setup(MersenneTwister(1), layer)
    stops = Corleone.get_timestops(layer, ps, st)
    @test Corleone.shooting_blocks(layer, ps, st, stops) == [1:4]
    actual, _ = layer(nothing, ps, st)
    expected, _ = with_ensemble(layer, EnsembleSerial())(nothing, ps, st)
    @test actual.u ≈ expected.u
    @test actual.residuals ≈ expected.residuals
end

@testset "Private mutable lookup caches" begin
    ic = Corleone.InitialCondition([1.0, 2.0], 1.0; cache = Dict{Float64, Int}())
    _, st = LuxCore.setup(MersenneTwister(1), ic)
    private = Corleone.private_shooting_state(ic, st)
    @test private.lookup.cache !== st.lookup.cache
    @test private.lookup.collection === st.lookup.collection
    private.lookup.cache[1.0] = 1
    @test isempty(st.lookup.cache)
end
