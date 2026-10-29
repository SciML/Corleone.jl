include("lotka_setup.jl")
oed = lotka_layer(; svd = true)
ps, st = LuxCore.setup(StableRNG(1), oed)
p = ComponentArray(ps)
lb, ub = Corleone.get_bounds(oed, ps, st) .|> ComponentArray

sol, _ = oed(nothing, p, st)

@test sol.u[1] == vcat(u0, zeros(7))
@test size.(sensitivities(oed, sol))[1] == (2, 2) # 2 singular values to describe F
@test collect(lb) == zeros(length(cgrid) + length(tgrid1) + length(tgrid2))
@test collect(ub) == ones(length(cgrid) + length(tgrid1) + length(tgrid2))

res = zeros(3)
@test_nowarn CorleoneOED.get_sampling_sums!(res, oed, nothing, ps, st)
@test [12.0, 12.0, 0.0] == res
@test CorleoneOED.get_sampling_sums(oed, nothing, ps, st) == res[1:2]
@test_nowarn @inferred CorleoneOED.get_sampling_sums(oed, nothing, ps, st)


optprob = OptimizationProblem(oed, ACriterion(), M = [4.0, 4.0])

@test optprob.f(optprob.u0, optprob.p) ≈ 0.05352869250783344

uopt = solve(
    optprob, Ipopt.Optimizer(),
    tol = 1.0e-6,
    hessian_approximation = "limited-memory", # sb="yes",
    max_iter = 100,
    print_level = 0,
)

@testset "Solution" begin
    @test uopt.objective ≈ 0.0370750919
    @test uopt.retcode == SciMLBase.ReturnCode.Success
    opt_p = zero(p) .+ uopt
    sol, _ = oed(nothing, opt_p, st)
    c1 = parameter_samples(sol, :p₁)
    u1 = parameter_samples(sol, :w1)
    u2 = parameter_samples(sol, :w2)
    @test sol.u[1][1:2] == u0
    accurate = accurate_prediction(oed, opt_p, st)
    G_reference, F_reference = reference_lotka_information(oed, accurate)
    @test sensitivities(oed, accurate)[end] ≈ G_reference * oed.basis
    F = fisher_information(oed, sol)
    @test fisher_information(oed, accurate) ≈ oed.basis' * F_reference * oed.basis
    @test ACriterion()(F) ≈ ACriterion()([88.89549299146556 13.63402689709759; 13.63402689709759 41.72266803144628])
    @test sol.t[c1 .> 0.1] == [0.0, 0.15, 0.25, 0.3, 0.45, 0.5, 0.6, 0.75, 0.9, 1.0, 4.8, 4.95, 5.0, 5.1, 5.25, 5.4, 5.5, 5.55, 5.7, 5.75, 5.85, 6.0, 6.15, 6.25, 6.3, 6.45, 6.5, 6.6, 6.75, 6.9, 7.0, 7.05, 7.2, 7.25, 7.35, 7.5, 7.65, 7.75]
    @test sol.t[u1 .> 0.1] == [2.55, 2.7, 2.75, 2.85, 3.0, 3.15, 3.25, 3.3, 3.45, 3.5, 3.6, 3.75, 3.9, 4.0, 4.05, 4.2, 4.25, 4.35, 4.5, 4.65, 4.75, 4.8, 4.95, 5.0, 10.65, 10.75, 10.8, 10.95, 11.0, 11.1, 11.25, 11.4, 11.5, 11.55, 11.7, 11.75, 12.0]
    @test sol.t[u2 .> 0.1] == [3.0, 3.15, 3.25, 3.3, 3.45, 3.5, 3.6, 3.75, 3.9, 4.0, 4.05, 4.2, 4.25, 4.35, 4.5, 4.65, 4.75, 4.8, 4.95, 5.0, 5.1, 5.25, 5.4, 10.65, 10.75, 10.8, 10.95, 11.0, 11.1, 11.25, 11.4, 11.5, 11.55, 11.7, 11.75, 12.0]
end
