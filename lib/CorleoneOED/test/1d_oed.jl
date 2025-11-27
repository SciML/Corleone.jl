using CorleoneOED
using OrdinaryDiffEqTsit5
using Test
using StableRNGs
using LuxCore
using ComponentArrays
using Optimization, OptimizationMOI, Ipopt

rng = StableRNG(1111)

function lin_dyn(u, p, t)
  return [p[1] * u[1]]
end

u0 = [1.0]
tspan = (0.0, 1.0)
p = [-2.0]

prob = ODEProblem(lin_dyn, u0, tspan, p,)
ol = SingleShootingLayer(prob, Tsit5(), controls=[], bounds_p=([-2.0], [-2.0]))
ps, st = LuxCore.setup(rng, ol)
traj_ref, _ = ol(nothing, ps, st)

oed = OEDLayer{false}(
  ol,
  params=[1,],
  measurements=[ControlParameter(collect(0.0:0.1:0.9), controls=ones(10), bounds=(0.0, 1.0)),],
  observed=(u, p, t) -> [u[1]]
)

ps, st = LuxCore.setup(rng, oed)
lb, ub = Corleone.get_bounds(oed)

@test_nowarn @inferred oed(nothing, ps, st)
traj_oed, _ = oed(nothing, ps, st)

@test traj_oed.t == 0.0:0.1:1.0
@test CorleoneOED.__fisher_information(oed, traj_oed)[end] == first(CorleoneOED.fisher_information(oed, nothing, ps, st))
@test reduce(vcat, first(CorleoneOED.observed_equations(oed, nothing, ps, st))) == reduce(vcat, first.(traj_oed.u))

@test LuxCore.parameterlength(oed) == LuxCore.parameterlength(ol) + 10
@test lb.p == ub.p == [-2.0]
@test all(iszero, lb.controls)
@test all(isone, ub.controls)

@testset "Criteria" begin
  foreach(
    (ACriterion(), DCriterion(), ECriterion(),
    FisherACriterion(), FisherDCriterion(), FisherECriterion())
  ) do crit
    @test_nowarn @inferred crit(oed, nothing, ps, st)
  end
end

function optimize_1d(oed, ps, st, crit)
  p = ComponentArray(ps)

  objective = let ax = getaxes(p), crit = crit, oed = oed
    (p, st) -> begin
      ps = ComponentArray(p, ax)
      first(crit(oed, nothing, ps, st))
    end
  end

  sampling_cons = let ax = getaxes(p)
    (res, p, ::Any) -> begin
      ps = ComponentArray(p, ax)
      res[1] = sum(ps.controls)
    end
  end

  optfun = OptimizationFunction(
    objective, AutoForwardDiff(), cons=sampling_cons
  )

  optprob = OptimizationProblem(
    optfun, collect(p), st, lb=reduce(vcat, collect(lb)), ub=reduce(vcat, collect(ub)),
    lcons=[0.0], ucons=[2.0]
  )

  uopt = solve(optprob, Ipopt.Optimizer(),
    tol=1e-10,
    hessian_approximation="limited-memory",
    max_iter=300,
    print_level=1,
  )
end


@testset "Optimization" begin
  for MEASUREMENT in (true, false)
    oed = OEDLayer{MEASUREMENT}(
      ol,
      params=[1,],
      measurements=[ControlParameter(collect(0.0:0.1:0.9), controls=ones(10), bounds=(0.0, 1.0)),],
      observed=(u, p, t) -> [u[1]]
    )
    ps, st = LuxCore.setup(rng, oed)
    # Check for the extra grid 
    if MEASUREMENT
      @test hasfield(typeof(st), :observation_grid)
    end
    uref = if MEASUREMENT
      vcat(-2.0, zeros(5), [1.0, 1.0], zeros(3))
    else
      vcat(-2.0, zeros(4), [1.0, 1.0], zeros(4))
    end
    for crit in (ACriterion(), DCriterion(), ECriterion())
      p_opt = optimize_1d(oed, ps, st, crit)
      @test SciMLBase.successful_retcode(p_opt)
      @test isapprox(collect(p_opt), uref)
    end
  end
end

#==
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
==#
