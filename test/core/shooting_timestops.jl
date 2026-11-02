using Corleone, LuxCore, OrdinaryDiffEqTsit5, Random, Test

@testset "Setup caches schedules without initializing tunable values" begin
    calls = zeros(Int, 4)
    function initializer(i)
        (rng, n::Int) -> begin
            calls[i] += 1
            rand(rng, n)
        end
    end
    prob = ODEProblem((u, p, t) -> p, [1.0, 2.0], (0.0, 2.0), [1.0, 2.0])
    layer = ShootingLayer(
        prob, Tsit5(),
        Tunable(1; kind = Val(:variable), default = initializer(1)) => [-1.0, 1.0, 3.0],
        Constant(2; kind = Val(:variable), default = initializer(3)) => [1.5],
        Tunable(1; kind = Val(:parameter), default = initializer(2)) => [0.5, 1.5],
        Constant(2; kind = Val(:parameter), default = initializer(4)) => nothing
    )
    st = LuxCore.initialstates(MersenneTwister(42), layer)
    @test calls == [0, 0, 1, 1]
    @test st.timestops == [0.0, 0.5, 1.0, 1.5, 2.0]
    @test st.timestops == Corleone.get_timestops(layer, nothing, st)
    fill!(calls, 0)
    rng = MersenneTwister(42)
    expected_rng = copy(rng)
    ps, st = LuxCore.setup(rng, layer)
    @test calls == ones(Int, 4)
    # Setup draws tunable IC values, tunable parameters, then the two fixed sets.
    expected_ic = rand(expected_rng, 3)
    expected_p = rand(expected_rng, 3)
    expected_fixed_ic = rand(expected_rng, 1)
    expected_fixed_p = rand(expected_rng, 1)
    @test ps.initialcondition.initialconditions.u₁.parameter.ps == expected_ic
    @test ps.parameter.parameters.p₁.parameter.ps == expected_p
    @test st.initialcondition.initialconditions.u₂.parameters.parameter.ps == expected_fixed_ic
    @test st.parameter.parameters.p₂.parameters.parameter.ps == expected_fixed_p
    @test rand(rng) == rand(expected_rng)
    trajectory, next_st = layer(nothing, ps, st)
    @test calls == ones(Int, 4)
    @test next_st.timestops === st.timestops
    @test trajectory.st === next_st
    @test trajectory.st.timestops === st.timestops
end

@testset "Cached grids survive serial, threaded, and fallback state handoff" begin
    prob = ODEProblem(
        (u, p, t) -> [p[1]], [1.0], (0.0, 2.0), [2.0];
        saveat = [0.25, 1.25]
    )
    for alg in (EnsembleSerial(), EnsembleThreads()), resets in (false, true)
        definitions = resets ? (Tunable(1; kind = Val(:variable), default = [10.0]) => [1.0],) : ()
        layer = ShootingLayer(prob, Tsit5(), definitions...; ensemblealg = alg)
        ps, st = LuxCore.setup(MersenneTwister(1), layer)
        @test st.timestops == (resets ? [0.0, 1.0, 2.0] : [0.0, 2.0])
        trajectory, next_st = layer(nothing, ps, st)
        @test next_st.timestops === st.timestops
        @test trajectory.st === next_st
        @test 0.25 in trajectory.t && 1.25 in trajectory.t
        @test last(trajectory.u) ≈ (resets ? [12.0] : [5.0])
        repeated, last_st = layer(nothing, ps, next_st)
        @test repeated.u ≈ trajectory.u
        @test repeated.t == trajectory.t
        @test repeated.residuals ≈ trajectory.residuals
        @test last_st.timestops === st.timestops
    end
end

@testset "Explicit grid overrides and parameter schedule refresh" begin
    prob = ODEProblem((u, p, t) -> [p[1]], [1.0], (0.0, 2.0), [2.0])
    layer = ShootingLayer(prob, Tsit5(), 1 => [1.0])
    ps, st = LuxCore.setup(MersenneTwister(1), layer)
    ps.parameter.parameters.p₁.parameter.ps .= [2.0, 4.0]
    @test st.timestops == [0.0, 1.0, 2.0]
    custom = [0.0, 0.5, 1.0, 2.0]
    trajectory, next_st = layer((prob, custom), ps, st)
    @test trajectory.t == custom
    @test next_st.timestops === custom
    @test st.timestops == [0.0, 1.0, 2.0]
    repeated, again = layer(nothing, ps, next_st)
    @test repeated.t == custom
    @test repeated.u ≈ trajectory.u
    @test again.timestops === custom

    # Changing a lookup grid requires explicitly refreshing the cached boundaries.
    st.parameter.parameters.p₁.lookup.collection[1] = 0.75
    refreshed = merge(st, (; timestops = Corleone.get_timestops(layer, nothing, st)))
    @test refreshed.timestops == [0.0, 0.75, 2.0]
    @test st.timestops == [0.0, 1.0, 2.0]
    trajectory, updated = layer(nothing, ps, refreshed)
    @test trajectory.t == [0.0, 0.75, 2.0]
    @test last(trajectory.u) ≈ [7.5]
    @test updated.timestops === refreshed.timestops
end
