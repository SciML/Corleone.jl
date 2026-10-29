using ForwardDiff
include("lotka_setup.jl")
oed = lotka_layer()
ps, st = LuxCore.setup(StableRNG(1), oed)
lb, ub = get_bounds(oed, ps, st) .|> ComponentArray
sol, _ = oed(nothing, ps, st)
@test sol.u[1] == vcat(u0, zeros(7))
@test all(iszero, lb)
@test all(isone, ub)
res = zeros(3)
@test_nowarn sampling_sums!(res, oed, nothing, ps, st)
@test res == [12.0, 12.0, 0.0]
@test sampling_sums(oed, nothing, ps, st) == res[1:2]
@test_nowarn @inferred sampling_sums(oed, nothing, ps, st)
@testset "Integer measurements" begin
    for discrete in (true, false), integer_sampling in (true, false), fixed in (true, false), shooting in (true, false)
        design = lotka_layer(; discrete, fixed, shooting)
        opt = OptimizationProblem(design, ACriterion(); M = [4.0, 4.0], integer_weights = integer_sampling)
        @test (isnothing(opt.int) ? 0 : sum(opt.int)) == (integer_sampling ? length(tgrid1) + length(tgrid2) : 0)
        if discrete && integer_sampling && !fixed && !shooting
            ps, st = LuxCore.setup(StableRNG(1), design)
            packed = OptimizationProblem(
                design, ACriterion();
                ps = ComponentArray(ps), st, M = [4.0, 4.0], integer_weights = true
            )
            @test packed.int == opt.int
            @test packed.u0 == opt.u0
        end
    end
end
@testset "Single experiments" begin
    for shooting in (false, true)
        _oed = lotka_layer(; shooting)
        _ps, _st = LuxCore.setup(StableRNG(1), _oed)
        _p = ComponentArray(_ps)
        _optprob = OptimizationProblem(_oed, ACriterion(); ps = _ps, st = _st, M = [4.0, 4.0])
        @test _optprob.f(_optprob.u0, _optprob.p) ≈ (shooting ? 0.6199466255548527 : 0.05352869250783344)
        _uopt = solve(_optprob, Ipopt.Optimizer(); tol = 1.0e-6, hessian_approximation = "limited-memory", max_iter = 100, print_level = 0)
        @testset "Solution" begin
            @test _uopt.objective ≈ 0.03707508955468313
            @test _uopt.retcode == SciMLBase.ReturnCode.Success
            opt_p = zero(_p) .+ _uopt
            sol, _ = _oed(nothing, opt_p, _st)
            c1 = parameter_samples(sol, :p₁)
            u1 = parameter_samples(sol, :w1)
            u2 = parameter_samples(sol, :w2)
            @test sol.u[1][1:2] == u0
            @test isapprox(sensitivities(_oed, sol)[end], [0.0224954223439133 -1.2367919565412857; -4.44473522371497 -2.9641449776773956], atol = 1.0e-6)
            F, _ = fisher_information(_oed, nothing, opt_p, _st)
            accurate = accurate_prediction(_oed, opt_p, _st)
            G_reference, F_reference = reference_lotka_information(_oed, accurate)
            @test fisher_information(_oed, accurate) ≈ F_reference
            @test sensitivities(_oed, accurate)[end] ≈ G_reference
            @test ACriterion()(F) ≈ ACriterion()([38.58695777364362 5.304118558316029; 5.304118558316029 92.03122059902915])
            @test unique(sol.t[c1 .> 0.1]) == [0.0, 0.15, 0.25, 0.3, 0.45, 0.5, 0.6, 0.75, 0.9, 1.0, 4.8, 4.95, 5.0, 5.1, 5.25, 5.4, 5.5, 5.55, 5.7, 5.75, 5.85, 6.0, 6.15, 6.25, 6.3, 6.45, 6.5, 6.6, 6.75, 6.9, 7.0, 7.05, 7.2, 7.25, 7.35, 7.5, 7.65, 7.75]
            @test unique(sol.t[u1 .> 0.1]) == [2.55, 2.7, 2.75, 2.85, 3.0, 3.15, 3.25, 3.3, 3.45, 3.5, 3.6, 3.75, 3.9, 4.0, 4.05, 4.2, 4.25, 4.35, 4.5, 4.65, 4.75, 4.8, 4.95, 5.0, 10.65, 10.75, 10.8, 10.95, 11.0, 11.1, 11.25, 11.4, 11.5, 11.55, 11.7, 11.75, 12.0]
            @test unique(sol.t[u2 .> 0.1]) == [3.0, 3.15, 3.25, 3.3, 3.45, 3.5, 3.6, 3.75, 3.9, 4.0, 4.05, 4.2, 4.25, 4.35, 4.5, 4.65, 4.75, 4.8, 4.95, 5.0, 5.1, 5.25, 5.4, 10.65, 10.75, 10.8, 10.95, 11.0, 11.1, 11.25, 11.4, 11.5, 11.55, 11.7, 11.75, 12.0]
        end
    end
end
@testset "Multiple experiments" begin
    for discrete in (false, true), split in (false, true), fixed in (false, true), shooting in (false, true)
        grids = (0.0:0.25:11.75, 0.0:0.25:11.75)
        first_layer = lotka_layer(; discrete, fixed, shooting, grids)
        second_layer = lotka_layer(; discrete, fixed, shooting, grids, params = split ? [3] : [2, 3])
        multi = MultiExperimentLayer((first_layer, second_layer))
        ps, st = LuxCore.setup(StableRNG(1), multi)
        trajectories, next_st = @inferred multi(nothing, ps, st)
        F, _ = @inferred fisher_information(multi, nothing, ps, st)
        expected = sum(zip(values(multi.experiments), values(trajectories), multi.embeddings)) do (layer, traj, E)
            E * fisher_information(layer, traj; add_initial = false) * E'
        end
        @test F ≈ expected
        packed = ComponentArray(ps)
        @test_nowarn @inferred multi(nothing, packed, st)
        @test first(@inferred fisher_information(multi, nothing, packed, st)) ≈ F
        @test sampling_sums(multi, nothing, packed, st) == sampling_sums(multi, nothing, ps, st)
        @test get_number_of_shooting_constraints(multi, packed, st) == get_number_of_shooting_constraints(multi, ps, st)
        @test keys(next_st.experiments) == keys(st.experiments)
        sums = @inferred sampling_sums(multi, nothing, ps, st)
        @test sums == (discrete ? fill(48.0, 4) : fill(12.0, 4))
        res = zeros(4)
        @test_nowarn @inferred sampling_sums!(res, multi, nothing, ps, st)
        @test res == sums
        lb, ub = get_bounds(multi, ps, st) .|> ComponentArray
        @test length(lb) == length(ub) == LuxCore.parameterlength(multi)
        @test all(lb .<= ub)
        @test length(shooting_constraints(multi, trajectories)) == get_number_of_shooting_constraints(multi, ps, st)
        prior = update_fim(multi, [(; ps, st)], st)
        @test prior.F_init ≈ F
        if fixed && !shooting && !split
            q, reconstruct = pack(default_vectorizer(), ps)
            accurate_problem = remake(first_layer.augmented_prob; abstol = 1.0e-12, reltol = 1.0e-12)
            loss(q) = tr(first(fisher_information(multi, accurate_problem, reconstruct(q), st)))
            gradient = ForwardDiff.gradient(loss, q)
            @test all(isfinite, gradient)
            @test dot(gradient, q) ≈ loss(q)
        end
        @test first(fisher_information(multi, nothing, ps, prior)) ≈ 2F
        @test_nowarn OptimizationProblem(multi, ACriterion(); ps, st, M = fill(discrete ? 12.0 : 4.0, 4))
        if !shooting && !fixed && !split
            integer_problem = OptimizationProblem(
                multi, ACriterion(); ps, st,
                M = fill(discrete ? 12.0 : 4.0, 4), integer_weights = true
            )
            @test sum(integer_problem.int) == 4length(grids[1])
            packed_integer = OptimizationProblem(
                multi, ACriterion();
                ps = ComponentArray(ps), st, M = fill(discrete ? 12.0 : 4.0, 4), integer_weights = true
            )
            @test packed_integer.int == integer_problem.int
        end
    end
end
