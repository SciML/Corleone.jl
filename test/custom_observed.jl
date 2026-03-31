# Tests for custom observed functions feature
# Custom observed functions allow defining arbitrary functions (u, p, t) -> value
# that can be accessed symbolically through SymbolicIndexingInterface

using Test
using Corleone
using LuxCore
using OrdinaryDiffEqTsit5
using Random
using SymbolicIndexingInterface

rng = MersenneTwister(42)

# Simple dynamics for testing
function simple_dynamics!(du, u, p, t)
    du[1] = -u[1] + p[1]
    du[2] = u[1] * u[2]
    return nothing
end

@testset "Custom Observed Functions" begin
    @testset "Basic construction with custom_observed" begin
        # Setup a simple trajectory
        u = [[1.0, 0.5], [0.9, 0.55], [0.8, 0.6]]
        p = [0.1]
        t = [0.0, 0.1, 0.2]
        sys = SymbolCache([:x, :y], [:p], :t)
        controls = (;)  # empty controls
        shooting = nothing

        # Create trajectory with custom observed functions
        traj = Trajectory(sys, u, p, t, controls, shooting;
            sum_state = (u, p, t) -> u[1] + u[2],
            double_p = (u, p, t) -> 2 * p[1],
        )

        @test traj.custom_observed isa NamedTuple
        @test haskey(traj.custom_observed, :sum_state)
        @test haskey(traj.custom_observed, :double_p)
        @test :sum_state in keys(traj.custom_observed)
        @test :double_p in keys(traj.custom_observed)
    end

    @testset "is_observed for custom functions" begin
        u = [[1.0, 0.5], [0.9, 0.55]]
        p = [0.1]
        t = [0.0, 0.1]
        sys = SymbolCache([:x, :y], [:p], :t)
        controls = (;)
        shooting = nothing

        traj = Trajectory(sys, u, p, t, controls, shooting;
            custom_func = (u, p, t) -> u[1]^2,
        )

        # Custom observed should be recognized
        @test SymbolicIndexingInterface.is_observed(traj, :custom_func) == true
        # Regular states are not "observed" in this context
        @test SymbolicIndexingInterface.is_observed(traj, :x) == false
        # Non-existent symbols should return false
        @test SymbolicIndexingInterface.is_observed(traj, :nonexistent) == false
    end

    @testset "observed returns correct values" begin
        u = [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]
        p = [5.0]
        t = [0.0, 1.0, 2.0]
        sys = SymbolCache([:x, :y], [:p], :t)
        controls = (;)
        shooting = nothing

        traj = Trajectory(sys, u, p, t, controls, shooting;
            sum_vals = (u, p, t) -> u[1] + u[2],
            prod_vals = (u, p, t) -> u[1] * u[2],
        )

        # Test using getsym which is the proper interface
        @testset "sum_vals observed function" begin
            f_obs = getsym(traj, :sum_vals)
            vals = f_obs(traj)
            @test vals isa AbstractVector
            @test length(vals) == length(t)
            @test vals[1] ≈ 1.0 + 2.0  # u[1] + u[2] at first point
            @test vals[2] ≈ 2.0 + 3.0
            @test vals[3] ≈ 3.0 + 4.0
        end

        @testset "prod_vals observed function" begin
            f_obs = getsym(traj, :prod_vals)
            vals = f_obs(traj)
            @test length(vals) == length(t)
            @test vals[1] ≈ 1.0 * 2.0
            @test vals[2] ≈ 2.0 * 3.0
            @test vals[3] ≈ 3.0 * 4.0
        end
    end

    @testset "Multiple custom observed functions" begin
        u = [[1.0, 2.0, 3.0]]
        p = [10.0]
        t = [0.0]
        sys = SymbolCache([:x, :y, :z], [:p], :t)
        controls = (;)
        shooting = nothing

        traj = Trajectory(sys, u, p, t, controls, shooting;
            norm = (u, p, t) -> sqrt(sum(u.^2)),
            energy = (u, p, t) -> 0.5 * sum(u.^2),
            momentum = (u, p, t) -> u[1] + 2*u[2] + 3*u[3],
            param_scaled = (u, p, t) -> p[1] * (u[1] + u[2]),
        )

        @test length(keys(traj.custom_observed)) == 4

        # Test each one using getsym
        norm_val = getsym(traj, :norm)(traj)[1]
        @test norm_val ≈ sqrt(1.0^2 + 2.0^2 + 3.0^2)

        energy_val = getsym(traj, :energy)(traj)[1]
        @test energy_val ≈ 0.5 * (1.0^2 + 2.0^2 + 3.0^2)

        momentum_val = getsym(traj, :momentum)(traj)[1]
        @test momentum_val ≈ 1.0 + 2*2.0 + 3*3.0

        param_val = getsym(traj, :param_scaled)(traj)[1]
        @test param_val ≈ 10.0 * (1.0 + 2.0)
    end

    @testset "Empty custom_observed (default)" begin
        u = [[1.0]]
        p = [1.0]
        t = [0.0]
        sys = SymbolCache([:x], [:p], :t)
        controls = (;)
        shooting = nothing

        # Without custom_observed keyword
        traj = Trajectory(sys, u, p, t, controls, shooting)

        @test traj.custom_observed isa NamedTuple
        @test isempty(traj.custom_observed)
        @test SymbolicIndexingInterface.is_observed(traj, :anything) == false
    end

    @testset "Custom observed with controls" begin
        # Integration test with actual controls
        prob = ODEProblem(simple_dynamics!, [1.0, 0.5], (0.0, 1.0), [0.1])
        ctrl = ControlParameter([0.0, 0.5, 1.0]; name = :u, controls = (rng, t) -> zeros(length(t)))

        layer = SingleShootingLayer(prob, ctrl; algorithm = Tsit5())
        ps, st = LuxCore.setup(rng, layer)
        traj, _ = layer(nothing, ps, st)

        # Add custom observed after the fact would require reconstruction
        # Instead, we test that the default trajectory works
        @test traj.custom_observed isa NamedTuple
        @test isempty(traj.custom_observed)

        # Control should still be accessible
        @test SymbolicIndexingInterface.is_observed(traj, :u) == true
    end

    @testset "Function signature correctness" begin
        # Ensure functions receive correct (u, p, t) arguments - state vector, params, scalar time
        u = [[1.0], [2.0]]
        p = [10.0]
        t = [0.0, 1.0]
        sys = SymbolCache([:x], [:p], :t)
        controls = (;)
        shooting = nothing

        # Track what arguments the function receives
        received_calls = Vector{Tuple{Vector{Float64}, Vector{Float64}, Float64}}()

        traj = Trajectory(sys, u, p, t, controls, shooting;
            inspector = (state, params, time) -> begin
                push!(received_calls, (copy(state), copy(params), time))
                return 0.0
            end,
        )

        # Call via getsym to trigger the function
        vals = getsym(traj, :inspector)(traj)
        @test length(vals) == length(t)
        @test length(received_calls) == 2

        # Check that the function was called with correct arguments at each timepoint
        @test received_calls[1][1] == [1.0]  # state at t=0.0
        @test received_calls[1][2] == [10.0]  # params
        @test received_calls[1][3] == 0.0   # time

        @test received_calls[2][1] == [2.0]  # state at t=1.0
        @test received_calls[2][2] == [10.0]  # params
        @test received_calls[2][3] == 1.0   # time
    end

    @testset "Type stability with NamedTuple" begin
        u = [[1.0]]
        p = [1.0]
        t = [0.0]
        sys = SymbolCache([:x], [:p], :t)
        controls = (;)
        shooting = nothing

        traj = Trajectory(sys, u, p, t, controls, shooting;
            a = (u, p, t) -> 1.0,
            b = (u, p, t) -> 2.0,
        )

        # NamedTuple should preserve order and allow iteration
        @test keys(traj.custom_observed) == (:a, :b)
        @test traj.custom_observed.a isa Function
        @test traj.custom_observed.b isa Function
    end
end