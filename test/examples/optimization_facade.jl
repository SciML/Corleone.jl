using Test, Corleone, LuxCore, OrdinaryDiffEqTsit5, ComponentArrays, Random
using Optimization, OptimizationMOI, Ipopt, ForwardDiff

@testset "Optimization bridge and automatic shooting constraints" begin
    prob = ODEProblem((u, p, t) -> [p[1]], [0.0], (0.0, 2.0), [0.2]; saveat = [0.5, 1.5], abstol = 1.0e-11, reltol = 1.0e-11)
    layer = ShootingLayer(
        prob, Tsit5(),
        Tunable(1; kind = Val(:variable), default = 0.2, lowerbound = 0.0, upperbound = 2.0) => [1.0],
        Tunable(1; kind = Val(:parameter), default = 0.2, lowerbound = 0.0, upperbound = 1.0) => nothing
    )
    ps, st = LuxCore.setup(MersenneTwister(1), layer)
    ps, st = add_parameter_segments!(layer, ps, st)
    d = CorleoneDynamicOptProblem(
        layer, traj -> (last(traj.u)[1] - 1)^2,
        :u₁ => (; t = [0.5, 1.0, 1.5, 2.0], bounds = (0.0, 1.5)); ps, st
    )
    @test Base.get_extension(Corleone, :CorleoneComponentArraysExtension) !== nothing
    opt = OptimizationProblem(d, AutoForwardDiff(); function_kwargs = (;))
    @test opt.p === st
    @test opt.u0 isa Vector{Float64}
    @test opt.u0 == collect(ComponentArray(ps))
    @test opt.lb == zero(opt.u0)
    @test opt.ub == collect(ComponentArray(last(get_bounds(layer, ps, st))))
    @test opt.lcons == d.lcons && opt.ucons == d.ucons
    @test length(opt.lcons) == 6 # Four path constraints, one state and one parameter equality.
    q = collect(opt.u0)
    @test (@inferred opt.f.f(q, opt.p)) ≈ optimization_objective(d)
    vals = similar(q, length(d.lcons))
    opt.f.cons(vals, q, opt.p)
    @test vals ≈ optimization_constraints(d)
    @test ForwardDiff.gradient(x -> opt.f.f(x, opt.p), q) ≈ [2 * (0.4 - 1), 0.0, 2 * (0.4 - 1)]
    @test all(
        isfinite, ForwardDiff.jacobian(
            x -> begin
                out = similar(x, length(d.lcons)); opt.f.cons(out, x, opt.p); out
            end, q
        )
    )
    # Explicit structured parameters retain their backing storage; no setup is repeated.
    structured = ComponentArray(ps)
    opt2 = OptimizationProblem(d, AutoForwardDiff(); vectorize = Val(:ComponentArrays), ps = structured)
    @test opt2.u0 === getdata(structured)
    @test opt2.u0 == opt.u0
    allocating = OptimizationFunction(d, AutoForwardDiff(); inplace = false)
    @test allocating.cons(q, st) ≈ vals
    @test_throws ArgumentError OptimizationFunction(d, AutoForwardDiff(); cons = (args...) -> nothing)
    shooting_only = CorleoneDynamicOptProblem(layer, :u₁; ps, st)
    shooting_opt = OptimizationProblem(shooting_only, AutoForwardDiff())
    @test shooting_opt.lcons == shooting_opt.ucons == zeros(2)
    @test shooting_opt.f.cons !== nothing
    result = solve(
        opt, Ipopt.Optimizer(); max_iter = 100, tol = 1.0e-8,
        hessian_approximation = "limited-memory", print_level = 0
    )
    @test Corleone.SciMLBase.successful_retcode(result)
    @test result.objective < 1.0e-10
    _, reconstruct = pack(Val(:ComponentArrays), ps)
    optimized = reconstruct(result.u)
    residuals = optimization_constraints(d, optimized, st)
    @test maximum(abs, residuals[(end - 1):end]) < 1.0e-7
    @test all(residuals[1:4] .>= -1.0e-7) && all(residuals[1:4] .<= 1.5 + 1.0e-7)

    # The convenience form omits constraint callbacks and bounds when unconstrained.
    simple = ShootingLayer(ODEProblem((u, p, t) -> [p[1]], [0.0], (0.0, 1.0), [0.3]), Tsit5())
    unconstrained = OptimizationProblem(simple, AutoForwardDiff(); loss = traj -> (last(traj.u)[1] - 0.6)^2)
    @test unconstrained.f.cons === nothing
    @test unconstrained.lcons === unconstrained.ucons === nothing
    @test isfinite(unconstrained.f(unconstrained.u0, unconstrained.p))
end
