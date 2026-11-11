# A failed integration segment must not crash reverse-mode differentiation: the pad
# (`pad_failed_segment!`) mutates the solution arrays with `push!`, which Zygote cannot
# trace. An explicit rrule keeps the gradient path alive and hands the solution
# cotangent back to the solver's pullback in the format the solver expects. The unit
# check exercises the rrule's cotangent rebuild directly; the layer check only verifies
# that differentiating through a padded failed segment runs.
using Corleone, LuxCore, OrdinaryDiffEqTsit5, Random, RecursiveArrayTools, Test
using SciMLBase, SciMLSensitivity, Zygote

@testset "Failed-segment pad differentiation" begin
    f! = (du, u, p, t) -> (du[1] = p[1] * u[1])
    # maxiters = 0: every segment solve fails before the first step, so only the start
    # point is saved and the pad must repeat it at the segment end.
    prob = ODEProblem(
        f!, [1.0], (0.0, 1.0), [0.05];
        maxiters = 0, verbose = false, sensealg = SciMLSensitivity.ForwardDiffSensitivity(),
    )
    stage = ShootingLayer(prob, Tsit5(), Corleone.Tunable(1; kind = Val(:parameter)) => nothing)
    ps, st = LuxCore.setup(Random.default_rng(), stage)

    traj, _ = stage(nothing, ps, st)
    @test traj.t == [0.0, 1.0]
    @test traj.u == [[1.0], [1.0]]

    # The gradient runs to completion (no mutation error) and is finite. The solve of a
    # failed segment carries no sensitivity, so the values are zero; running through is
    # the regression check.
    g = Zygote.gradient(ps) do ps
        traj, _ = stage(nothing, ps, st)
        return last(traj.u)[1]
    end
    @test g isa Tuple
    @test all(isfinite, collect(float.(g[1].parameter.parameters.p₁.parameter.ps)))
end

@testset "Pad rrule cotangent rebuild" begin
    # Real failed solve: only the start point is saved, the pad repeats it at the
    # segment end, and the solution cotangent flows back in the DiffEqArray format
    # Zygote produces for SciML solutions.
    prob = ODEProblem(
        (du, u, p, t) -> (du[1] = p[1] * u[1]), [1.0], (0.0, 1.0), [0.05];
        maxiters = 0, verbose = false, sensealg = SciMLSensitivity.ForwardDiffSensitivity(),
    )
    sol = solve(prob, Tsit5())
    @test !SciMLBase.successful_retcode(sol)
    padded, pullback = Corleone.ChainRulesCore.rrule(
        Corleone.pad_failed_segment!, sol, 0.0, 1.0, ()
    )
    @test padded.t == sol.t
    @test padded.u == sol.u
    Δsol = RecursiveArrayTools.DiffEqArray([[0.0], [1.0]], [0.0, 1.0])
    Δsol_args, Δprim, Δt0, Δt1, Δexpected = pullback(Δsol)
    @test Δsol_args isa Corleone.ChainRulesCore.NoTangent
    @test collect(Δprim.u) == [[1.0]]
    @test collect(Δprim.t) == [0.0]
    @test Δt1 == 1.0
    @test Δt0 isa Corleone.ChainRulesCore.ZeroTangent
    @test Δexpected isa Corleone.ChainRulesCore.ZeroTangent

    # Struct cotangents (NamedTuple primal mock): the padded entries accumulate into
    # the repeated source state and the pushed-time cotangent routes to `t1`.
    mock = (; t = [0.0], u = [[1.0]], retcode = SciMLBase.ReturnCode.Failure)
    padded2, pullback2 = Corleone.ChainRulesCore.rrule(
        Corleone.pad_failed_segment!, mock, 0.0, 1.0, ()
    )
    @test padded2.t == [0.0, 1.0]
    @test padded2.u == [[1.0], [1.0]]
    Δmock = Corleone.ChainRulesCore.Tangent{typeof(mock)}(; u = [[0.0], [1.0]], t = [0.0, 1.0])
    Δmock_args, Δmock_cot, _, Δmock_t1, _ = pullback2(Δmock)
    @test Δmock_args isa Corleone.ChainRulesCore.NoTangent
    @test Δmock_cot.u == [[1.0]]
    @test Δmock_cot.t == [0.0]
    @test Δmock_t1 == 1.0

    # A failed retcode within (t0, t1) that still saved steps pads the missing end
    # only, including every problem-level save time the segment did not reach.
    mock3 = (; t = [0.0, 0.5], u = [[1.0], [2.0]], retcode = SciMLBase.ReturnCode.Failure)
    padded3 = Corleone.pad_failed_segment!(mock3, 0.0, 1.0, [0.75])
    @test padded3.t == [0.0, 0.5, 0.75, 1.0]
    @test padded3.u == [[1.0], [2.0], [2.0], [2.0]]
end
