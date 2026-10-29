using Corleone, BenchmarkTools
using OrdinaryDiffEqTsit5, StableRNGs, LuxCore

const SUITE = BenchmarkGroup()
const rng = StableRNG(42)

function lotka_dynamics!(du, u, p, t)
    du[1] = u[1] - p[2] * prod(u[1:2]) - 0.4 * p[1] * u[1]
    du[2] = -u[2] + p[3] * prod(u[1:2]) - 0.2 * p[1] * u[2]
    return du[3] = (u[1] - 1.0)^2 + (u[2] - 1.0)^2
end

tspan = (0.0, 12.0)
u0 = [0.5, 0.7, 0.0]
p0 = [0.0, 1.0, 1.0]

prob = ODEProblem(
    lotka_dynamics!, u0, tspan, p0; abstol = 1.0e-8, reltol = 1.0e-6
)
cgrid = collect(0.0:0.1:11.9)
N = length(cgrid)
# AirspeedVelocity runs this same script against both the PR and its base.
# Keep the old API branch until main also provides the annotation-based layers.
if isdefined(Corleone, :PiecewiseConstant)
    control = Corleone.PiecewiseConstant(cgrid[2:end], zeros(N); name = :fishing)
    SUITE["control"] = BenchmarkGroup()
    SUITE["control"]["construct"] = @benchmarkable Corleone.PiecewiseConstant(
        $(cgrid[2:end]), zeros($N); name = :fishing
    )
    SUITE["control"]["setup"] = @benchmarkable LuxCore.setup($rng, $control)
    cps, cst = LuxCore.setup(rng, control)
    SUITE["control"]["lookup"] = @benchmarkable $control(1.25, $cps, $cst)

    control_definition = Tunable(1; default = zeros(N), lowerbound = 0.0, upperbound = 1.0) => cgrid[2:end]
    resets = map(1:3) do i
        Tunable(i; kind = Val(:variable), default = u0[i], lowerbound = 0.0, upperbound = 100.0) => [3.0, 6.0, 9.0]
    end
    SUITE["layer"] = BenchmarkGroup()
    SUITE["layer"]["single_shooting"] = @benchmarkable ShootingLayer(
        $prob, Tsit5(), $control_definition; fixed_parameters = [2, 3]
    )
    SUITE["layer"]["multiple_shooting"] = @benchmarkable ShootingLayer(
        $prob, Tsit5(), $control_definition, $(resets)...; fixed_parameters = [2, 3]
    )
    layer = ShootingLayer(prob, Tsit5(), control_definition; fixed_parameters = [2, 3])
    mlayer = ShootingLayer(prob, Tsit5(), control_definition, resets...; fixed_parameters = [2, 3])
else
    control = ControlParameter(
        cgrid; name = :fishing, bounds = (0.0, 1.0), controls = zeros(N)
    )
    SUITE["control"] = BenchmarkGroup()
    SUITE["control"]["construct"] = @benchmarkable ControlParameter(
        $cgrid; name = :fishing, bounds = (0.0, 1.0), controls = zeros($N)
    )
    SUITE["control"]["check_consistency"] = @benchmarkable Corleone.check_consistency($rng, $control)
    SUITE["control"]["get_controls"] = @benchmarkable Corleone.get_controls($rng, $control)

    SUITE["layer"] = BenchmarkGroup()
    SUITE["layer"]["single_shooting"] = @benchmarkable SingleShootingLayer(
        $prob, Tsit5(); controls = (1 => $control,), bounds_p = ([1.0, 1.0], [1.0, 1.0])
    )
    SUITE["layer"]["multiple_shooting"] = @benchmarkable MultipleShootingLayer(
        $prob, Tsit5(), 0.0, 3.0, 6.0, 9.0; controls = (1 => $control,),
        bounds_ic = ([0.1, 0.1, 0.0], [100.0, 100.0, 100.0]),
        bounds_p = ([1.0, 1.0], [1.0, 1.0])
    )
    layer = SingleShootingLayer(
        prob, Tsit5(); controls = (1 => control,), bounds_p = ([1.0, 1.0], [1.0, 1.0])
    )
    mlayer = MultipleShootingLayer(
        prob, Tsit5(), 0.0, 3.0, 6.0, 9.0; controls = (1 => control,),
        bounds_ic = ([0.1, 0.1, 0.0], [100.0, 100.0, 100.0]),
        bounds_p = ([1.0, 1.0], [1.0, 1.0])
    )
end

ps, st = LuxCore.setup(rng, layer)
mps, mst = LuxCore.setup(rng, mlayer)
SUITE["layer"]["setup"] = @benchmarkable LuxCore.setup($rng, $layer)
SUITE["layer"]["forward"] = @benchmarkable $layer(nothing, $ps, $st)
SUITE["layer"]["ms_forward"] = @benchmarkable $mlayer(nothing, $mps, $mst)
