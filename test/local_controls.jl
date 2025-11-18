using Corleone
using OrdinaryDiffEqTsit5
using Test
using Random
using LuxCore

rng = Random.default_rng()

c = ControlParameter(0:.01:1.0)
lb, ub = Corleone.get_bounds(c)

@test c.controls === Corleone.default_u
@test c.bounds === Corleone.default_bounds
@test c.t == collect(0:.01:1.0)
@test_nowarn Corleone.check_consistency(rng, c)
@test unique(lb) == [-Inf]
@test unique(ub) == [Inf]

c1 = ControlParameter(1.:10., bounds = (0.,1.))
lb1, ub1 = Corleone.get_bounds(c1)
@test unique(lb1) == [0.0]
@test unique(ub1) == [1.0]
@test_nowarn Corleone.check_consistency(rng, c1)

c2 = ControlParameter(1.:10., bounds = (-ones(10), ones(10)))
lb2, ub2 = Corleone.get_bounds(c2)
@test unique(lb2) == [-1.0]
@test unique(ub2) == [1.0]
@test_nowarn Corleone.check_consistency(rng, c2)

c3 = ControlParameter(1.:10., bounds = (-ones(10), ones(10)), controls = collect(0.0:0.1:0.9))
@test Corleone.get_controls(rng, c3) == collect(0.0:0.1:0.9)
@test_nowarn Corleone.check_consistency(rng, c3)

c4 = ControlParameter(1.:10., bounds = (-ones(11), ones(10)), controls = collect(0.0:0.1:0.9))
@test_throws "Incompatible control bound definition" Corleone.check_consistency(rng, c4)

c5 = ControlParameter(1.:10., bounds = (-ones(10), ones(10)), controls = collect(0.0:0.1:1.0))
@test_throws "Sizes are inconsistent" Corleone.check_consistency(rng, c5)

c5 = ControlParameter(1.:10., bounds = (ones(10), -ones(10)), controls = collect(0.0:0.1:1.0))
@test_throws "Bounds are inconsistent" Corleone.check_consistency(rng, c5)
