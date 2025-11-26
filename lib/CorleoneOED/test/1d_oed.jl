using CorleoneOED
using OrdinaryDiffEqTsit5
using Test
using StableRNGs 

using ComponentArrays
using Optimization, OptimizationMOI, Ipopt

rng = Random.default_rng()

function lin_dyn(u, p, t)
  return [p[1] * u[1]]
end

u0 = [1.0]
tspan = (0.0, 1.0)
p = [-2.0, 0.0]


prob = ODEProblem(lin_dyn2, u0, tspan, p,)

ol = SingleShootingLayer(prob, Tsit5(), controls=[1 => ControlParameter([0.0, 0.1, 0.2]), 2 => ControlParameter([0.0, 0.5],)])

oed = CorleoneOED.OEDLayer{false}(ol, params=[1,],)

oed = CorleoneOED.OEDLayer{true}(
  ol,
  params=[1, 2],
  #measurements = [ControlParameter([0., 0.25, 0.5, 0.75], controls = ones(4)), ControlParameter([0.25, 0.75], controls = ones(2))], 
	observed = (u,p,t) -> [u[1], (u[1]+p[2])^2] 
)

LuxCore.initialstates(rng, ol)

LuxCore.initialstates(rng, oed)

ps, st = LuxCore.setup(Random.default_rng(), oed)



p = ComponentArray(ps)
lb, ub = Corleone.get_bounds(ol)
crit = ACriterion()
ACrit = crit(ol)

sampling_cons = let ax = getaxes(p), sampling = Corleone.get_sampling_constraint(ol)
  (res, p, ::Any) -> begin
    ps = ComponentArray(p, ax)
    res .= sampling(ps, nothing)
  end
end

optfun = OptimizationFunction(
  ACrit, AutoForwardDiff(), cons=sampling_cons
)

optprob = OptimizationProblem(
  optfun, collect(p), lb=collect(lb), ub=collect(ub), lcons=zeros(1), ucons=[0.2]
)

uopt = solve(optprob, Ipopt.Optimizer(),
  tol=1e-10,
  hessian_approximation="limited-memory",
  max_iter=300
)
@testset "Convergence" begin
  @test SciMLBase.successful_retcode(uopt)
end

@testset "Information gain: Optimality criteria" begin
  IG = InformationGain(ol, uopt.u)

  multiplier = uopt.original.inner.mult_g

  optimality = tr.(IG.global_information_gain[1])

  idxs_opt = optimality .> multiplier[1]
  t_optimality = IG.t[idxs_opt]
  @test !isempty(t_optimality)
  @test all(0.4 .<= t_optimality .<= 0.6)

  wopt = (uopt + zero(p)).controls

  idxs_w = wopt .> 0.99

  t_sampling = ol.layer.controls[1].t[idxs_w]

  @test all(0.4 .<= t_sampling .<= 0.6)
end
