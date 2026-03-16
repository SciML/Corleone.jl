using Corleone
using LuxCore
using OrdinaryDiffEqTsit5
using Random
using SymbolicIndexingInterface
using Test

rng = MersenneTwister(11)

function observed_dynamics!(du, u, p, t)
    uctrl = p[1]
    du[1] = -0.6 * u[1] + uctrl
    du[2] = u[1] - 0.2 * u[2]
    return nothing
end

@testset "Observed helper utilities" begin
    @test Corleone._extract_timepoints(2.5) == [2.5]
    @test Corleone._extract_timepoints(:( [1.0, 3.5] )) == [1.0, 3.5]
    @test_throws AssertionError Corleone._extract_timepoints(:(x(1.0)))

    collector = Dict(:x => Float64[], :u => Float64[], :y => Float64[])
    ex = :(x(2.0) + sin(u([1.0, 4.0])) + cos(z(7.0)))
    Corleone._collect_timepoints!(collector, ex)

    @test collector[:x] == [2.0]
    @test collector[:u] == [1.0, 4.0]
    @test isempty(collector[:y])

    # Non-expression input should be ignored by the fallback method.
    @test Corleone._collect_timepoints!(collector, 3.14) === nothing

    idx = Dict(1.0 => 2, 2.0 => 5, 4.0 => 8)
    @test Corleone._extract_timeindex(2.0, idx) == 5
    @test Corleone._extract_timeindex(:( [1.0, 4.0] ), idx) == [2, 8]
    @test_throws AssertionError Corleone._extract_timeindex(:(x(1.0)), idx)

    replacer = Dict(:x => Dict(1.0 => 2, 3.0 => 7), :u => Dict(2.0 => 5))
    replaced = Corleone.replace_timepoints(:(x([1.0, 3.0]) + u(2.0)), replacer)
    @test replaced isa Expr
    @test replaced.head == :call
    @test replaced.args[1] == :+
    @test replaced.args[2] isa Expr
    @test replaced.args[2].args[1] == :getindex
    @test replaced.args[2].args[2] == :x
    @test replaced.args[2].args[3] == [2, 7]
    @test replaced.args[3] isa Expr
    @test replaced.args[3].args[1] == :getindex
    @test replaced.args[3].args[2] == :u
    @test replaced.args[3].args[3] == 5

    @test Corleone.replace_timepoints(42, replacer) == 42
    @test Corleone.replace_timepoints(:(1 + 2), replacer) == :(1 + 2)

    grid = [0.0, 1.0, 2.0, 5.0]
    points = [-1.0, 0.4, 1.7, 6.0]
    inds = Corleone.find_indices(points, grid)

    @test inds[-1.0] == 1
    @test inds[0.4] == 1
    @test inds[1.7] == 2
    @test inds[6.0] == 4
end

@testset "ObservedLayer constructor and forward pass" begin
    sys = SymbolCache([:x, :y], [:u], :t)
    prob = ODEProblem(
        ODEFunction(observed_dynamics!; sys = sys),
        [1.2, -0.4],
        (0.0, 10.0),
        [0.0],
        saveat = [4.0, 5.0, 6.0],
    )

    controls = (
        ControlParameter(
            [0.0, 2.0, 4.0, 6.0, 8.0];
            name = :u,
            controls = (rng, t) -> fill(0.25, length(t)),
            bounds = t -> (fill(-2.0, length(t)), fill(2.0, length(t))),
        ),
    )

    shooting = SingleShootingLayer(prob, controls...; algorithm = Tsit5(), name = :obs_single)

    expr1 = :(x(3.0) - u(3.5))
    expr2 = :(sum(x([1.0, 3.0])) + y(4.25))
    expr3 = :(x(10.0) - x(0.0))

    observed = Corleone.ObservedLayer(shooting, expr1, expr2, expr3; name = :obs_layer)

    saveat = get(Corleone.get_problem(observed.layer).kwargs, :saveat, Float64[])
    @test issorted(saveat)
    @test allunique(saveat)
    @test 3.0 in saveat
    @test 3.5 in saveat
    @test 4.25 in saveat

    ps, st = LuxCore.setup(rng, observed)
    output, st2 = observed(nothing, ps, st)

    @test hasproperty(output, :observations)
    @test hasproperty(output, :trajectory)
    @test st2 == st

    traj = output.trajectory
    obsvals = output.observations

    @test length(obsvals) == 3

    xvals = getsym(traj, :x)(traj)
    yvals = getsym(traj, :y)(traj)
    uvals = getsym(traj, :u)(traj)

    idx_x = Corleone.find_indices([3.0, 1.0, 3.0, 10.0, 0.0], traj.t)
    idx_u = Corleone.find_indices([3.5], traj.t)
    idx_y = Corleone.find_indices([4.25], traj.t)

    expected1 = xvals[idx_x[3.0]] - uvals[idx_u[3.5]]
    expected2 = sum(xvals[[idx_x[1.0], idx_x[3.0]]]) + yvals[idx_y[4.25]]
    expected3 = xvals[idx_x[10.0]] - xvals[idx_x[0.0]]

    @test isapprox(obsvals[1], expected1; atol = 1.0e-10)
    @test isapprox(obsvals[2], expected2; atol = 1.0e-10)
    @test isapprox(obsvals[3], expected3; atol = 1.0e-10)

    forward_only = @inferred first(observed(nothing, ps, st))
    @test forward_only.observations == obsvals
end
