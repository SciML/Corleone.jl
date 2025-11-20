using Corleone
using OrdinaryDiffEqTsit5
using Test
using Random
using LuxCore
using ComponentArrays
using Optimization, OptimizationMOI, Ipopt

using SciMLSensitivity
using SciMLSensitivity.ReverseDiff
using SciMLSensitivity.Zygote
using SymbolicIndexingInterface

rng = Random.default_rng()

function lotka_dynamics!(du, u, p, t)
  du[1] = u[1] - p[2] * prod(u[1:2]) - 0.4 * p[1] * u[1]
  du[2] = -u[2] + p[3] * prod(u[1:2]) - 0.2 * p[1] * u[2]
  du[3] = (u[1] - 1.0)^2 + (u[2] - 1.0)^2
  return
end

tspan = (0.0, 12.0)
u0 = [0.5, 0.7, 0.0]
p0 = [0.0, 1.0, 1.0]
prob = ODEProblem(lotka_dynamics!, u0, tspan, p0; abstol=1e-8, reltol=1e-6)

cgrid = collect(0.0:0.1:11.9)
N = length(cgrid)
control = ControlParameter(
  cgrid, name=:fishing, bounds=(0.0, 1.0), controls=zeros(N)
)

layer = SingleShootingLayer(prob, Tsit5(); controls=(1 => control,), bounds_p=([1.0, 1.0], [1.0, 1.0]))

ps, st = LuxCore.setup(rng, layer)

sol, _ = layer(nothing, ps, st)

@test sol.t == getsym(sol, :t)(sol)
@test sol.p[1] == getsym(sol, :p₁)(sol)
@test sol.p[2] == getsym(sol, :p₂)(sol)

x = reduce(hcat, sol.u)

for (i, sym) in enumerate((:x₁, :x₂, :x₃, :u₁))
  getter = getsym(sol, sym)
	@test getter(sol) == x[i, :]
end

@test_nowarn @inferred layer(nothing, ps, st)

@test allunique(sol.t)
@test LuxCore.parameterlength(layer) == N + 2


for AD in (AutoForwardDiff(), AutoReverseDiff(), AutoZygote())
  prob = ODEProblem(lotka_dynamics!, u0, tspan, p0; abstol=1e-8, reltol=1e-6, sensealg=AD == AutoZygote() ? ForwardDiffSensitivity() : SciMLBase.NoAD())

  cgrid = collect(0.0:0.1:11.9)
  N = length(cgrid)
  control = ControlParameter(
    cgrid, name=:fishing, bounds=(0.0, 1.0), controls=zeros(N)
  )

  layer = SingleShootingLayer(prob, Tsit5(); controls=(1 => control,), bounds_p=([1.0, 1.0], [1.0, 1.0]))

  ps, st = LuxCore.setup(rng, layer)

  p = ComponentArray(ps)
  lb, ub = Corleone.get_bounds(layer)

  @test lb.p == ub.p == p0[2:end]
  @test lb.controls == zeros(N)
  @test ub.controls == ones(N)
  @test size(p, 1) == LuxCore.parameterlength(layer)

  objective = let layer = layer, ax = getaxes(p), st = st
    (p, ::Any) -> begin
      ps = ComponentArray(p, ax)
      sol, _ = layer(nothing, ps, st)
      last(sol.u)[3]
    end
  end

  @test isapprox(objective(p, nothing), 6.062277454291031, atol=1e-4)

  optfun = OptimizationFunction(objective, AD)
  optprob = OptimizationProblem(optfun, collect(p), lb=reduce(vcat, collect(lb)), ub=reduce(vcat, collect(ub)))

  sol = solve(optprob, Ipopt.Optimizer(), max_iter=1000, tol=5e-6,
    hessian_approximation="limited-memory")

  @test SciMLBase.successful_retcode(sol)
  @test isapprox(sol.objective, 1.344336, atol=1e-4)

  p_opt = sol.u .+ zero(p)

  @test isempty(p_opt.u0)
  @test p_opt.p == p0[2:end]
end

