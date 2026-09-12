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
control = ControlParameter(
    cgrid; name = :fishing, bounds = (0.0, 1.0), controls = zeros(N)
)

# =============================================================================
# ControlParameter construction and checks
# =============================================================================

SUITE["control"] = BenchmarkGroup()

SUITE["control"]["construct"] = @benchmarkable ControlParameter(
    $cgrid; name = :fishing, bounds = (0.0, 1.0), controls = zeros($N)
)
SUITE["control"]["check_consistency"] = @benchmarkable Corleone.check_consistency(
    $rng, $control
)
SUITE["control"]["get_controls"] = @benchmarkable Corleone.get_controls(
    $rng, $control
)

# =============================================================================
# Shooting layers
# =============================================================================

SUITE["layer"] = BenchmarkGroup()

SUITE["layer"]["single_shooting"] = @benchmarkable SingleShootingLayer(
    $prob, Tsit5(); controls = (1 => $control,),
    bounds_p = ([1.0, 1.0], [1.0, 1.0])
)
SUITE["layer"]["multiple_shooting"] = @benchmarkable MultipleShootingLayer(
    $prob, Tsit5(), 0.0, 3.0, 6.0, 9.0; controls = (1 => $control,),
    bounds_ic = ([0.1, 0.1, 0.0], [100.0, 100.0, 100.0]),
    bounds_p = ([1.0, 1.0], [1.0, 1.0])
)

layer = SingleShootingLayer(
    prob, Tsit5(); controls = (1 => control,), bounds_p = ([1.0, 1.0], [1.0, 1.0])
)
ps, st = LuxCore.setup(rng, layer)

SUITE["layer"]["setup"] = @benchmarkable LuxCore.setup($rng, $layer)
SUITE["layer"]["forward"] = @benchmarkable $layer(nothing, $ps, $st)

mlayer = MultipleShootingLayer(
    prob, Tsit5(), 0.0, 3.0, 6.0, 9.0; controls = (1 => control,),
    bounds_ic = ([0.1, 0.1, 0.0], [100.0, 100.0, 100.0]),
    bounds_p = ([1.0, 1.0], [1.0, 1.0])
)
mps, mst = LuxCore.setup(rng, mlayer)

SUITE["layer"]["ms_forward"] = @benchmarkable $mlayer(nothing, $mps, $mst)
