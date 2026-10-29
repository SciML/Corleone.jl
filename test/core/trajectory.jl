using Corleone, LuxCore, OrdinaryDiffEqTsit5, Random, Test
import SymbolicIndexingInterface as SII
import SciMLBase

@testset "Trajectory assembly and implicit parameters" begin
    prob = ODEProblem((u, p, t) -> [p[1]], [1.0], (0.0, 2.0), [2.0])
    layer = ShootingLayer(prob, Tsit5(), 1 => [1.0])
    ps, st = LuxCore.setup(MersenneTwister(1), layer)
    ps.parameter.parameters.p₁.parameter.ps .= [2.0, 4.0]
    traj, next_st = layer(nothing, ps, st)
    @test traj isa Trajectory
    @test next_st isa NamedTuple
    @test traj.layer === layer
    @test traj.ps === ps
    @test traj.st === st
    @test traj.t == [0.0, 1.0, 2.0]
    @test traj.u ≈ [[1.0], [3.0], [7.0]]
    @test isempty(traj.reset_indices)
    @test all(isempty, traj.residuals)
    @test SII.state_values(traj) === traj.u
    @test SII.current_time(traj) === traj.t
    @test traj[:u₁] ≈ [1.0, 3.0, 7.0]
    @test traj[:t] == traj.t
    @test traj[:p₁] == [2.0, 4.0]
    @test SII.getp(traj, :p₁)(traj, 2) == 4.0
    @test SII.parameter_timeseries(traj, 1) == [0.0, 1.0]
    @test SII.parameter_values(traj) == [4.0]
    @test SII.parameter_values_at_time(traj, traj, 0.5) == [2.0]
    @test SII.parameter_values_at_time(traj, traj, 1.0) == [4.0]
    @test SII.parameter_values_at_time(traj, traj, 2.0) == [4.0]
    @test_throws DomainError SII.parameter_values_at_time(traj, traj, -0.1)

    @static if isdefined(SciMLBase, :DespecializedParameters)
        params = SciMLBase.DespecializedParameters([2.0])
        updated = SII.with_updated_parameter_timeseries_values(traj, params, 1 => [4.0])
        @test updated isa SciMLBase.DespecializedParameters
        @test updated.params == [4.0]
        @test SII.with_updated_parameter_timeseries_values(traj, params).params === params.params
        @test_throws ArgumentError SII.with_updated_parameter_timeseries_values(traj, params, 2 => [4.0])
    end

    # Parameter histories are implicit in the supplied Lux parameters and state.
    ps.parameter.parameters.p₁.parameter.ps .= 99.0
    st.parameter.parameters.p₁.lookup.collection[1] = 0.25
    @test traj[:p₁] == [99.0, 99.0]
    @test SII.parameter_timeseries(traj, 1) == [0.0, 0.25]
    @test SII.parameter_values_at_time(traj, traj, 0.5) == [99.0]
end

@testset "Reset boundaries survive reduction" begin
    prob = ODEProblem((u, p, t) -> [1.0], [1.0], (0.0, 3.0), [0.0])
    for fixed in (Int[], [1]), reset_value in (2.0, 5.0)
        # At t=1 the incoming value is 2, so one reset has exactly zero mismatch.
        layer = ShootingLayer(
            prob, Tsit5(), 1 => [2.0],
            (isempty(fixed) ? Tunable : Constant)(
                1;
                kind = Val(:variable), default = reset_value
            ) => [1.0],
        )
        ps, st = LuxCore.setup(MersenneTwister(1), layer)
        traj, _ = layer(nothing, ps, st)
        @test traj.t == [0.0, 1.0, 1.0, 2.0, 3.0]
        @test traj.reset_indices == [3]
        @test traj.u[2] ≈ [2.0]
        @test traj.u[3] == [reset_value]
        @test last(traj.u) ≈ [reset_value + 2]
        @test length(traj.residuals) == 3
        if isempty(fixed)
            @test traj.residuals[2] ≈ [reset_value - 2.0] atol = 1.0e-12
        else
            @test isempty(traj.residuals[2])
        end
        @test isempty(traj.residuals[3])
        # State-only boundaries do not add knots to the parameter history.
        @test SII.parameter_timeseries(traj, 1) == [0.0, 2.0]
    end
end

@testset "One segment and empty parameters" begin
    prob = ODEProblem((u, p, t) -> -u, [1.0], (0.0, 1.0), Float64[])
    layer = ShootingLayer(prob, Tsit5())
    ps, st = LuxCore.setup(MersenneTwister(1), layer)
    traj, _ = layer(nothing, ps, st)
    @test traj.t == [0.0, 1.0]
    @test length(traj.residuals) == 1
    @test isempty(SII.parameter_values(traj))
    @test last(traj.u) ≈ [exp(-1)] rtol = 1.0e-3
end

@testset "Fixed parameter schedules remain available" begin
    prob = ODEProblem((u, p, t) -> [p[1]], [1.0], (0.0, 2.0), [2.0])
    layer = ShootingLayer(prob, Tsit5(), 1 => [1.0]; fixed_parameters = [1])
    ps, st = LuxCore.setup(MersenneTwister(1), layer)
    st.parameter.parameters.p₁.parameters.parameter.ps .= [2.0, 4.0]
    traj, _ = layer(nothing, ps, st)
    @test traj.t == [0.0, 1.0, 2.0]
    @test last(traj.u) ≈ [7.0]
    @test traj[:p₁] == [2.0, 4.0]
end
