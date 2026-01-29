using Corleone
using OrdinaryDiffEqTsit5
using Test
using Random
using LuxCore
using ComponentArrays
using Optimization, OptimizationMOI, Ipopt

rng = Random.default_rng()

function lotka_dynamics!(du, u, p, t)
    du[1] = u[1] - p[2] * prod(u[1:2]) - 0.4 * p[1] * u[1]
    du[2] = -u[2] + p[3] * prod(u[1:2]) - 0.2 * p[1] * u[2]
    return du[3] = (u[1] - 1.0)^2 + (u[2] - 1.0)^2
end

tspan = (0.0, 12.0)
u0 = [0.5, 0.7, 0.0]
p0 = [0.0, 1.0, 1.0]

prob = ODEProblem(lotka_dynamics!, u0, tspan, p0; abstol = 1.0e-8, reltol = 1.0e-6)
cgrid = collect(0.0:0.1:11.9)
N = length(cgrid)
control = ControlParameter(
    cgrid, name = :fishing, bounds = (0.0, 1.0), controls = zeros(N)
)

layer = MultipleShootingLayer(prob, Tsit5(), 0.0, 3.0, 6.0, 9.0; controls = (1 => control,), bounds_ic = ([0.1, 0.1, 0.0], [100.0, 100.0, 100.0]), bounds_p = ([1.0, 1.0], [1.0, 1.0]))

ps, st = LuxCore.setup(rng, layer)
sol, _ = layer(nothing, ps, st)

@test_nowarn @inferred first(layer(nothing, ps, st))
@test_nowarn @inferred last(layer(nothing, ps, st))

@test allunique(sol.t)

p = ComponentArray(ps)
lb, ub = Corleone.get_bounds(layer) .|> ComponentArray

@test size(p, 1) == LuxCore.parameterlength(layer)

optprob = OptimizationProblem(layer, :x₃)

@test isapprox(optprob.f(optprob.u0, optprob.p), 1.2417260108009376, atol = 1.0e-4)

res = zeros(3 * 6)
@test isapprox(optprob.f.cons(res, p, st), [1.3757549609694821, 0.2235735751355118, 1.24172601080094, 0.0, 0.0, 1.375754960969481, 0.22357357513551102, 1.2417260108009385, 0.0, 0.0, 1.3757549609694824, 0.2235735751355129, 1.2417260108009414, 0.0, 0.0, 0.0, 0.0, 0.0])

sol = solve(
    optprob, Ipopt.Optimizer(), max_iter = 1000, tol = 1.0e-3,
    hessian_approximation = "limited-memory"
)

@test SciMLBase.successful_retcode(sol)
@test isapprox(sol.objective, 1.344336, atol = 1.0e-4)
