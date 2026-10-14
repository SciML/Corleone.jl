using CorleoneBase
using CommonSolve: CommonSolve, solve
using OrdinaryDiffEqTsit5: Tsit5
using SciMLBase: SciMLBase, ODEProblem, EnsembleSerial, remake, successful_retcode
using Test

@test realpath(pkgdir(CorleoneBase)) == realpath(joinpath(@__DIR__, "../.."))

@testset "ParallelProblem construction and interface" begin
    template = ODEProblem((u, p, t) -> -p .* u, [1.0], (0.0, 1.0), 0.5)
    sequential = SequentialProblem(template; transition = (seq, sol, i) -> seq.problem)
    parallel = ParallelProblem(sequential)

    @test parallel isa CorleoneBase.AbstractParallelProblem
    @test CorleoneBase.get_problem(parallel) === sequential
    # The default hook ignores the ensemble context and returns the template.
    @test CorleoneBase.prob_func(parallel, sequential, nothing) === sequential
end

# The implementation now supplies a multi-argument closure on Julia 1.10/1.11.
# Require the same successful ensemble behavior on every supported version.
@testset "Parallel supported-version contracts" begin
    @testset "Parallel ensemble workflow" begin
        template = ODEProblem((u, p, t) -> -p .* u, [1.0], (0.0, 1.0), 0.5)
        sequential = SequentialProblem(template; transition = (seq, sol, i) -> seq.problem)

        # Default hook: every trajectory solves the identical problem.
        parallel = ParallelProblem(sequential)
        ensemble = solve(
            parallel, Tsit5(), EnsembleSerial();
            trajectories = 4, save_everystep = false
        )
        @test ensemble isa SciMLBase.EnsembleSolution
        @test length(ensemble.u) == 4
        @test all(w -> w isa Vector, ensemble.u)
        @test all(w -> length(w) == 1, ensemble.u)
        @test all(w -> all(successful_retcode, w), ensemble.u)
        @test all(w -> w[1].prob.u0 == [1.0], ensemble.u)

        # The hook receives the ParallelProblem, the (copied) template, and a
        # SciMLBase.EnsembleContext whose sim_id selects the trajectory.
        seen = Tuple{Bool, Bool, Int}[]
        parallel2 = ParallelProblem(
            sequential; prob_func = (problem, tmpl, ctx) -> begin
                push!(seen, (problem === parallel2, tmpl isa SequentialProblem, ctx.sim_id))
                remake(tmpl; problem = remake(tmpl.problem; u0 = [Float64(ctx.sim_id)]))
            end
        )
        ensemble2 = solve(
            parallel2, Tsit5(), EnsembleSerial();
            trajectories = 3, save_everystep = false
        )
        @test length(ensemble2.u) == 3
        @test all(t -> t[1] && t[2], seen)
        @test sort([t[3] for t in seen]) == [1, 2, 3]
        @test [w[1].prob.u0[1] for w in ensemble2.u] == [1.0, 2.0, 3.0]

        # output_func is consumed by the EnsembleProblem, not forwarded to solve.
        outputs = solve(
            parallel, Tsit5(), EnsembleSerial();
            trajectories = 2, output_func = (sol, ctx) -> (length(sol), false)
        )
        @test length(outputs.u) == 2
        @test [outputs.u[i] for i in 1:length(outputs.u)] == [1, 1]

        # Stage-one saving is hard-coded, even with ordinary solver options.
        dense = solve(parallel, Tsit5(), EnsembleSerial(); trajectories = 1)
        sparse = solve(
            parallel, Tsit5(), EnsembleSerial();
            trajectories = 1, save_everystep = false
        )
        @test sparse.u[1][1].t == dense.u[1][1].t == [0.0, 1.0]
        remade = solve(parallel, Tsit5(), EnsembleSerial(); trajectories = 1, u0 = [2.0])
        @test remade.u[1][1].prob.u0 == [2.0]

        # A master `seed` is forwarded and makes per-trajectory draws
        # reproducible through ctx.rng.
        parallel3 = ParallelProblem(
            sequential; prob_func = (problem, tmpl, ctx) ->
            remake(tmpl; problem = remake(tmpl.problem; u0 = [rand(ctx.rng)]))
        )
        run_a = solve(
            parallel3, Tsit5(), EnsembleSerial();
            trajectories = 3, seed = 7, save_everystep = false
        )
        run_b = solve(
            parallel3, Tsit5(), EnsembleSerial();
            trajectories = 3, seed = 7, save_everystep = false
        )
        @test [w[1].prob.u0 for w in run_a.u] == [w[1].prob.u0 for w in run_b.u]

        # `trajectories` is a required keyword.
        @test_throws UndefKeywordError solve(parallel, Tsit5(), EnsembleSerial())
    end
end
