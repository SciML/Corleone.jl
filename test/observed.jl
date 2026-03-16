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

obs = ObservedExpressionLayer(:(sum(x([1., 2., 3., 4.])) - u(3.0) <= 0.0))


function flatten_grid(timestops)
    tgrid = Float64[]
    for bin in timestops
        for (t0, t1) in bin
            push!(tgrid, t0, t1)
        end
    end
    unique!(sort!(tgrid))
    return tgrid
end

@testset "ObservedExpressionLayer lowering" begin
    obs = ObservedExpressionLayer(:(x(10.0) - u(3.0) <= 0.0))
    @test obs.kind == :le
    @test obs.signal_names == (:x, :u)
    @test obs.timepoints == (x = [10.0], u = [3.0])
    @test obs.lowered == :(x[1] - u[1] - 0.0)

    obs_mix = ObservedExpressionLayer(:(sum(x([1.0, 2.0, 3.0, 4.0]) * y(4.25)) == 0.0))
    @test obs_mix.kind == :eq
    @test obs_mix.signal_names == (:x, :y)
    @test obs_mix.timepoints == (x = [1.0, 2.0, 3.0, 4.0], y = [4.25])
    @test obs_mix.lowered == :(sum(x[[1, 2, 3, 4]] * y[1]) - 0.0)
end

@testset "ObservedLayer setup and evaluation" begin
    sys = SymbolCache([:x, :y], [:u], :t)
    prob = ODEProblem(ODEFunction(observed_dynamics!; sys = sys), [1.2, -0.4], (0.0, 10.0), [0.0])

    controls = (
        ControlParameter(
            [0.0, 2.0, 4.0, 6.0, 8.0];
            name = :u,
            controls = (rng, t) -> fill(0.25, length(t)),
            bounds = t -> (fill(-2.0, length(t)), fill(2.0, length(t))),
        ),
    )

    shooting = SingleShootingLayer(prob, controls...; algorithm = Tsit5(), name = :obs_single)

    observed = ObservedLayer(
        shooting,
        (
            ineq = :(x(3.0) <= u(4.0)),
            eq = :(x(10.0) == u(3.0)),
            mix = :(sum(x([1.0, 2.0, 3.0, 4.0]) * y(4.25)) == 0.0),
        );
        name = :observed_layer,
    )

    saveat = get(Corleone.get_problem(observed.layer).kwargs, :saveat, Float64[])
    @test 4.25 in saveat
    @test 3.0 in saveat
    @test 10.0 in saveat

    ps, st = LuxCore.setup(rng, observed)
    vals, _ = observed(nothing, ps, st)
    traj, _ = observed.layer(nothing, ps.layer, st.layer)

    timegrid = flatten_grid(st.layer.timestops)
    xvals = getsym(traj, :x)(traj)
    yvals = getsym(traj, :y)(traj)
    uvals = getsym(traj, :u)(traj)

    idx_ineq = Corleone.find_time_indices((x = [3.0], u = [4.0]), st.layer.system, timegrid)
    expected_ineq = xvals[idx_ineq.x[1]] - uvals[idx_ineq.u[1]]
    @test isapprox(vals.ineq, expected_ineq; atol = 1.0e-12)

    idx_eq = Corleone.find_time_indices((x = [10.0], u = [3.0]), st.layer.system, timegrid)
    expected_eq = xvals[idx_eq.x[1]] - uvals[idx_eq.u[1]]
    @test isapprox(vals.eq, expected_eq; atol = 1.0e-12)

    idx_mix = Corleone.find_time_indices((x = [1.0, 2.0, 3.0, 4.0], y = [4.25]), st.layer.system, timegrid)
    expected_mix = sum(xvals[idx_mix.x] * yvals[idx_mix.y[1]])
    @test isapprox(vals.mix, expected_mix; atol = 1.0e-12)
end
