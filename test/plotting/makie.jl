using Test, Random, Corleone, LuxCore, OrdinaryDiffEqTsit5, CairoMakie
import CairoMakie: Makie
using SymbolicIndexingInterface

@testset "Makie reset segments and residuals" begin
    @test Base.get_extension(Corleone, :CorleoneMakieExtension) !== nothing
    prob = ODEProblem(
        (u, p, t) -> [1.0, 2.0, 3.0], [1.0, 10.0, 100.0], (0.0, 3.0), [0.5];
        saveat = [0.25, 0.75, 1.25, 1.75, 2.25, 2.75],
        tstops = [0.6, 1.6, 2.6], abstol = 1.0e-10, reltol = 1.0e-10
    )
    layer = ShootingLayer(
        prob, Tsit5(),
        Constant(1; kind = Val(:variable), default = [-4.0, 8.0]) => [1.0, 2.0],
        Tunable(2; kind = Val(:variable), default = [20.0, -5.0]) => [1.0, 2.5],
        Tunable(1; kind = Val(:parameter), default = 0.5) => [0.5, 1.5, 2.25]
    )
    ps, st = LuxCore.setup(MersenneTwister(1), layer)
    traj, _ = layer(nothing, ps, st)
    original_state = deepcopy(traj.st)
    specs = Makie.convert_arguments(Makie.Lines, traj)
    symbols = variable_symbols(symbolic_container(traj))

    @test traj.t[traj.reset_indices] == [1.0, 2.0, 2.5]
    @test length(traj.t) > length(traj.residuals) # Extra saved samples are not residual indices.
    span = only(filter(p -> p.type == :VSpan, specs))
    @test span.args[1] == [0.0, 2.0]
    @test span.args[2] == [1.0, 2.5]

    curves = filter(p -> p.type == :Lines, specs)
    @test length(curves) == 3
    for (curve, var) in zip(curves, symbols)
        points = only(curve.args)
        @test length(points) == length(traj.t) + length(traj.reset_indices)
        @test filter(p -> !any(isnan, p), points) == Makie.Point2f.(traj.t, traj[var])
        for (n, k) in enumerate(traj.reset_indices)
            gap = k + n - 1
            @test all(isnan, points[gap])
            @test points[gap - 1] == Makie.Point2f(traj.t[k - 1], traj[var][k - 1])
            @test points[gap + 1] == Makie.Point2f(traj.t[k], traj[var][k])
        end
    end

    # Only variable 2 is tunable: fixed resets and pass-through states have no bars.
    bars = only(filter(p -> p.type == :Rangebars, specs))
    @test bars.args[1] == [1.0, 2.5]
    @test bars.args[2] ≈ [12.0, -5.0] # Correct endpoints for upward and downward resets.
    @test bars.args[3] ≈ [20.0, 23.0]
    @test bars.kwargs[:color] == curves[2].kwargs[:color]
    @test bars.kwargs[:label] == "Δ$(symbols[2])"
    @test !haskey(bars.kwargs, :linestyle) || Makie.is_attribute(Makie.Rangebars, :linestyle)
    @test traj.st == original_state

    hidden = Makie.convert_arguments(Makie.Lines, traj; show_segments = false, show_variable_residuals = false)
    @test all(p -> p.type == :Lines, hidden)
    fixed = ShootingLayer(prob, Tsit5(), Constant(1; kind = Val(:variable)) => [1.0, 2.0])
    fixed_ps, fixed_st = LuxCore.setup(MersenneTwister(2), fixed)
    fixed_traj, _ = fixed(nothing, fixed_ps, fixed_st)
    @test !isempty(fixed_traj.reset_indices)
    @test all(isempty, fixed_traj.residuals)
    @test !any(p -> p.type == :Rangebars, Makie.convert_arguments(Makie.Lines, fixed_traj))

    continuous = ShootingLayer(prob, Tsit5())
    continuous_ps, continuous_st = LuxCore.setup(MersenneTwister(3), continuous)
    continuous_traj, _ = continuous(nothing, continuous_ps, continuous_st)
    continuous_specs = Makie.convert_arguments(Makie.Lines, continuous_traj)
    @test all(p -> p.type == :Lines, continuous_specs)
    @test all(p -> !any(point -> any(isnan, point), only(p.args)), continuous_specs)

    # Exercise actual plot construction and a headless backend render as well as specs.
    fig = Corleone.trajectoryplot(traj; figure = (; size = (1000, 650)))
    mktempdir() do dir
        path = joinpath(dir, "reset_residuals.png")
        save(path, fig)
        @test filesize(path) > 0
    end
end

@testset "Shared-grid residual bars respect fixed members" begin
    prob = ODEProblem((u, p, t) -> ones(3), zeros(3), (0.0, 3.0), [0.5])
    layer = ShootingLayer(
        prob, Tsit5(),
        Tunable([3, 1]; kind = Val(:variable), default = [0.3, 0.1]) => [1.0, 2.0],
        Constant(2; kind = Val(:variable)) => [1.0, 2.0]
    )
    ps, st = LuxCore.setup(MersenneTwister(1), layer)
    trajectory, _ = layer(nothing, ps, st)
    specs = Makie.convert_arguments(Makie.Lines, trajectory)
    bars = filter(p -> p.type == :Rangebars, specs)
    @test length(bars) == 2
    @test map(p -> p.kwargs[:label], bars) == ["Δu₁", "Δu₃"]
    @test all(p -> p.args[1] == [1.0, 2.0], bars)
end
