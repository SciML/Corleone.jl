using Corleone, LuxCore, OrdinaryDiffEqTsit5, Random, Test, ComponentArrays

@testset "Split parameters at full-state shooting boundaries" begin
    for ensemblealg in (EnsembleSerial(), EnsembleThreads())
        prob = ODEProblem((u, p, t) -> [p[1], p[2]], [0.0, 0.0], (0.0, 4.0), [2.0, 10.0, 7.0])
        layer = ShootingLayer(
            prob, Tsit5(),
            Tunable(1; kind = Val(:variable)) => [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            Tunable(2; kind = Val(:variable)) => [1.0, 2.0, 3.0, 4.0],
            Tunable(1; kind = Val(:parameter), default = 2.0) => nothing,
            Tunable(2; kind = Val(:parameter), default = [10.0, 20.0, 30.0]) => [1.0, 2.5],
            Constant(3; kind = Val(:parameter), default = [7.0, 8.0]) => [0.5];
            ensemblealg
        )
        ps, st = LuxCore.setup(MersenneTwister(1), layer)
        old_p2 = st.parameter.parameters.p₂
        p2_state = merge(old_p2, (; lookup = merge(old_p2.lookup, (; cache = Dict(2.0 => 2)))))
        st = merge(st, (; parameter = merge(st.parameter, (; parameters = merge(st.parameter.parameters, (; p₂ = p2_state))))))
        original, _ = layer(nothing, ps, st)
        old_cache = copy(st.parameter.parameters.p₂.lookup.cache)
        count = get_number_of_shooting_constraints(layer, ps, st)
        setup_size = LuxCore.parameterlength(layer)
        expanded, state = add_parameter_segments!(layer, ps, st)
        @test expanded.parameter.parameters.p₁.parameter.ps == fill(2.0, 4)
        @test state.parameter.parameters.p₁.lookup.collection == [1.0, 2.0, 3.0]
        @test state.parameter.parameters.p₁.lookup.inserted == [2, 3, 4]
        @test expanded.parameter.parameters.p₂.parameter.ps == [10.0, 20.0, 20.0, 30.0, 30.0]
        @test state.parameter.parameters.p₂.lookup.collection == [1.0, 2.0, 2.5, 3.0]
        @test state.parameter.parameters.p₂.lookup.inserted == [3, 5]
        @test isempty(state.parameter.parameters.p₂.lookup.cache)
        @test st.parameter.parameters.p₂.lookup.cache == old_cache
        @test state.parameter.parameters.p₃ === st.parameter.parameters.p₃
        @test expanded.initialcondition === ps.initialcondition
        @test ps.parameter.parameters.p₁.parameter.ps == [2.0]
        @test st.parameter.parameters.p₁.lookup.collection === nothing
        @test st.parameter.parameters.p₂.lookup.collection == [1.0, 2.5]
        @test isempty(st.parameter.parameters.p₂.lookup.inserted)
        @test layer.parameter.parameters.p₁.lookup.collection === nothing
        @test LuxCore.parameterlength(layer) == setup_size
        @test state.timestops == Corleone.get_timestops(layer, nothing, state)
        pc = layer.parameter.parameters.p₂
        for (t, expected) in ((0.0, false), (1.0, false), (2.0, true), (2.5, false), (3.0, true))
            ((idx, inserted), _) = LuxCore.apply(pc.lookup, t, expanded.parameter.parameters.p₂.lookup, state.parameter.parameters.p₂.lookup)
            @test inserted == expected
            @test idx == searchsortedlast(state.parameter.parameters.p₂.lookup.collection, t) + 1
        end
        repeated, repeated_state = add_parameter_segments!(layer, expanded, state)
        @test repeated == expanded
        @test repeated_state == state
        trajectory, _ = layer(nothing, expanded, state)
        @test trajectory.t == original.t
        @test trajectory.u ≈ original.u
        @test shooting_constraints(trajectory) ≈ vcat(shooting_constraints(original), zeros(5))
        @test get_number_of_shooting_constraints(layer, expanded, state) == count + 5
        lower, upper = get_bounds(layer, expanded, state)
        @test length(lower.parameter.parameters.p₁.parameter.ps) == 4
        @test length(upper.parameter.parameters.p₂.parameter.ps) == 5

        # Structured optimizer parameters need fresh axes after expansion too.
        array_ps, array_st = add_parameter_segments!(layer, ComponentArray(ps), st)
        @test array_ps isa ComponentVector
        @test array_ps == ComponentArray(expanded)
        @test array_st.parameter.parameters.p₁.lookup.inserted == [2, 3, 4]
        array_trajectory, _ = layer(nothing, array_ps, array_st)
        @test array_trajectory.u ≈ original.u
    end
end

@testset "Partial resets and existing parameter switches" begin
    prob = ODEProblem((u, p, t) -> p, [0.0, 0.0], (0.0, 3.0), [1.0, 2.0])
    partial = ShootingLayer(prob, Tsit5(), Tunable(1; kind = Val(:variable)) => [1.0])
    ps, st = LuxCore.setup(MersenneTwister(1), partial)
    same_ps, _ = add_parameter_segments!(partial, ps, st)
    @test same_ps === ps

    layer = ShootingLayer(
        prob, Tsit5(),
        Tunable(1; kind = Val(:variable)) => [1.0, 2.0],
        Constant(2; kind = Val(:variable)) => [1.0, 2.0],
        Tunable(1; kind = Val(:parameter)) => [1.0, 2.0],
        Tunable(2; kind = Val(:parameter)) => [2.0]
    )
    ps, st = LuxCore.setup(MersenneTwister(1), layer)
    expanded, state = add_parameter_segments!(layer, ps, st)
    @test isempty(state.parameter.parameters.p₁.lookup.inserted)
    @test expanded.parameter.parameters.p₁ === ps.parameter.parameters.p₁
    @test state.parameter.parameters.p₂.lookup.inserted == [2]
    @test expanded.parameter.parameters.p₂.parameter.ps == [2.0, 2.0, 2.0]

    # Insert an earlier full reset and retain/remap the previous inserted indices.
    names = keys(state.initialcondition.initialconditions)
    conditions = map(values(state.initialcondition.initialconditions)) do ic
        if hasproperty(ic, :lookup)
            merge(ic, (; lookup = merge(ic.lookup, (; collection = Dict(0.5 => 1, 1.0 => 2)))))
        else
            merge(ic, (; states = merge(ic.states, (; lookup = merge(ic.states.lookup, (; collection = Dict(0.5 => 1, 1.0 => 2)))))))
        end
    end
    moved = merge(state, (; initialcondition = merge(state.initialcondition, (; initialconditions = NamedTuple{names}(conditions)))))
    expanded_again, state_again = add_parameter_segments!(layer, expanded, moved)
    @test state_again.parameter.parameters.p₂.lookup.collection == [0.5, 1.0, 2.0]
    @test state_again.parameter.parameters.p₂.lookup.inserted == [2, 3]
    @test expanded_again.parameter.parameters.p₂.parameter.ps == fill(2.0, 4)
end

@testset "On-demand parameter constraints follow state residuals" begin
    prob = ODEProblem((u, p, t) -> [p[1], p[2]], [0.0, 0.0], (0.0, 3.0), [2.0, 10.0, 7.0])
    for ensemblealg in (EnsembleSerial(), EnsembleThreads())
        layer = ShootingLayer(
            prob, Tsit5(),
            Tunable(1; kind = Val(:variable)) => [1.0, 2.0],
            Tunable(2; kind = Val(:variable)) => [1.0, 2.0],
            Tunable(1; kind = Val(:parameter)) => nothing,
            Tunable(2; kind = Val(:parameter), default = [10.0, 20.0]) => [1.5],
            Constant(3; kind = Val(:parameter)) => [1.0, 2.0]; ensemblealg
        )
        ps, st = LuxCore.setup(MersenneTwister(1), layer)
        ps, st = add_parameter_segments!(layer, ps, st)
        ps.parameter.parameters.p₁.parameter.ps .= [2.0, 5.0, 9.0]
        ps.parameter.parameters.p₂.parameter.ps .= [10.0, 12.0, 20.0, 25.0]
        for parameters in (ps, ComponentArray(ps))
            traj, _ = layer(nothing, parameters, st)
            state_residuals = reduce(vcat, traj.residuals)
            expected = vcat(state_residuals, [3.0, 4.0, 2.0, 5.0])
            @test shooting_constraints(traj) ≈ expected
            @test get_number_of_shooting_constraints(layer, nothing, st) == length(expected) == 8
            buffer = fill(NaN, length(expected))
            @test shooting_constraints!(buffer, traj) === buffer
            @test buffer ≈ expected
            storage = fill(-99.0, length(expected) + 2)
            @test shooting_constraints!(view(storage, 2:(length(storage) - 1)), traj) ≈ expected
            @test first(storage) == last(storage) == -99.0
            wrong = fill(99.0, length(expected) - 1)
            @test_throws DimensionMismatch shooting_constraints!(wrong, traj)
            @test all(==(99.0), wrong)
            # Only parameter differences change on query: no solve is repeated.
            parameters.parameter.parameters.p₁.parameter.ps[2] = 6.0
            @test shooting_constraints(traj) ≈ vcat(state_residuals, [4.0, 3.0, 2.0, 5.0])
            @test reduce(vcat, traj.residuals) == state_residuals
            parameters.parameter.parameters.p₁.parameter.ps[2] = 5.0
        end
    end
end
