using Corleone, LuxCore, OrdinaryDiffEqTsit5, Random, Test, ComponentArrays

with_ensemble(layer, alg) = ShootingLayer(
    layer.name, layer.problem, layer.alg, alg,
    layer.initialcondition, layer.parameter
)

@testset "Independent shooting blocks" begin
    prob = ODEProblem(
        (u, p, t) -> [p[1] * u[1], u[1] - u[2]],
        [1.0, 2.0], (0.0, 4.0), [0.2]; abstol = 1.0e-10, reltol = 1.0e-10,
        saveat = [0.25, 1.25, 2.25, 3.25]
    )
    layer = ShootingLayer(
        prob, Tsit5(),
        Tunable(1; kind = Val(:variable)) => [1.0, 2.0, 3.0, 4.0],
        Constant(2; kind = Val(:variable)) => [2.0, 4.0],
        Tunable(1; kind = Val(:parameter)) => [0.5, 1.5, 2.5, 3.5]
    )
    ps, st = LuxCore.setup(MersenneTwister(1), layer)
    stops = Corleone.get_timestops(layer, ps, st)
    states = @inferred Corleone.shooting_states(layer, ps, st, stops)
    @test map(s -> s.timestops, states) == [stops[1:5], stops[5:9]]
    local_st = first(states)
    @test last(local_st.timestops) == first(last(states).timestops)
    @test local_st.initialcondition.active !== st.initialcondition.active
    @test local_st.initialcondition.active !== last(states).initialcondition.active
    @test local_st.parameter.parameters.p₁.lookup.inserted !==
        last(states).parameter.parameters.p₁.lookup.inserted
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
    changed_st = merge(
        st, (;
            initialcondition = merge(
                st.initialcondition,
                (; initialconditions = children)
            ),
        )
    )
    changed_stops = Corleone.get_timestops(threaded, ps, changed_st)
    changed_st = merge(changed_st, (; timestops = changed_stops))
    @test map(s -> s.timestops, Corleone.shooting_states(threaded, ps, changed_st, changed_stops)) ==
        [changed_stops[1:7], changed_stops[7:9]]
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
    layer = ShootingLayer(
        prob, Tsit5(),
        Tunable(1; kind = Val(:variable)) => [1.0, 2.0]; ensemblealg = EnsembleThreads()
    )
    ps, st = LuxCore.setup(MersenneTwister(1), layer)
    stops = Corleone.get_timestops(layer, ps, st)
    @test only(Corleone.shooting_states(layer, ps, st, stops)).timestops == stops
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

# Keep the return unannotated: inference must reach the scalar through the layer.
parallel_scalar(layer, ps, st) = last(first(layer(nothing, ps, st)).u)[1]

@testset "Private worker state preserves heterogeneous field types" begin
    prob = ODEProblem((u, p, t) -> [p[1], p[2], p[3]], zeros(3), (0.0, 3.0), [1.0, 2.0, 3.0])
    layer = ShootingLayer(
        prob, Tsit5(),
        Tunable(1; kind = Val(:variable)) => [1.0, 2.0],
        Constant(2; kind = Val(:variable)) => [1.0, 2.0],
        Tunable(1; kind = Val(:parameter)) => [0.5],
        Constant(2; kind = Val(:parameter)) => nothing,
        Constant(3; kind = Val(:parameter)) => [1.5]
    )
    ps, st = LuxCore.setup(MersenneTwister(1), layer)
    local_st = @inferred Corleone.private_shooting_state(layer, st)
    @test typeof(local_st) === typeof(st)
    @test local_st == st
    @test local_st.initialcondition.active !== st.initialcondition.active
    @test local_st.initialcondition.initialconditions.u₃ === st.initialcondition.initialconditions.u₃
    for name in keys(layer.parameter.parameters)
        child = st.parameter.parameters[name]
        private_child = local_st.parameter.parameters[name]
        if layer.parameter.parameters[name] isa Corleone.FixedLayer
            @test private_child.parameters === child.parameters
            @test private_child.states.lookup.inserted !== child.states.lookup.inserted
        else
            @test private_child.lookup.inserted !== child.lookup.inserted
            @test private_child.lookup.collection === child.lookup.collection
        end
    end
    # Cache dictionaries must retain their types while receiving private storage.
    first_st = st.parameter.parameters.p₁
    lookup = merge(first_st.lookup, (; cache = Dict(0.75 => 2)))
    parameters = merge(st.parameter.parameters, (; p₁ = merge(first_st, (; lookup))))
    cached_st = merge(st, (; parameter = merge(st.parameter, (; parameters))))
    copied = @inferred Corleone.private_shooting_state(layer, cached_st)
    @test copied.parameter.parameters.p₁.lookup.cache == Dict(0.75 => 2)
    @test copied.parameter.parameters.p₁.lookup.cache !== cached_st.parameter.parameters.p₁.lookup.cache

    empty_prob = ODEProblem((u, p, t) -> -u, [1.0], (0.0, 1.0), Float64[])
    empty_layer = ShootingLayer(empty_prob, Tsit5())
    _, empty_st = LuxCore.setup(MersenneTwister(1), empty_layer)
    @test (@inferred Corleone.private_shooting_state(empty_layer, empty_st)) == empty_st
end

@testset "Threaded Float64 objectives infer without scalar assertions" begin
    prob = ODEProblem((u, p, t) -> [p[1] * u[1] + p[2], p[3]], [1.0, 2.0], (0.0, 3.0), [0.2, 0.1, 0.5])
    threaded = ShootingLayer(
        prob, Tsit5(),
        Tunable(1; kind = Val(:variable)) => [1.0, 2.0],
        Constant(2; kind = Val(:variable)) => [1.0, 2.0],
        Tunable(1; kind = Val(:parameter)) => [0.5, 1.5],
        Constant(2; kind = Val(:parameter)) => nothing,
        Constant(3; kind = Val(:parameter)) => nothing;
        ensemblealg = EnsembleThreads()
    )
    ps, st = LuxCore.setup(MersenneTwister(1), threaded)
    serial = with_ensemble(threaded, EnsembleSerial())
    @test (@inferred parallel_scalar(threaded, ps, st)) ≈ parallel_scalar(serial, ps, st)
    # ComponentArray views are the parameter representation used by optimization.
    array_ps = ComponentArray(ps)
    @test (@inferred parallel_scalar(threaded, array_ps, st)) ≈ parallel_scalar(serial, array_ps, st)
end
