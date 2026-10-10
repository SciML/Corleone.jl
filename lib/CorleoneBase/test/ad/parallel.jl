using Distributed
using SciMLBase: EnsembleSerial, EnsembleThreads, EnsembleDistributed
using DifferentiationInterface: AutoForwardDiff, AutoReverseDiff
using ForwardDiff
using ReverseDiff
using SciMLSensitivity: SensitivityADPassThrough
using CorleoneBase
using Test

if Threads.nthreads() == 1
    # Pkg.test and CI may start a single-threaded driver. Run the real contracts
    # in a multithreaded subprocess rather than skipping threaded coverage.
    script = abspath(@__FILE__)
    command = `$(Base.julia_cmd()) --startup-file=no --threads=4 --project=$(Base.active_project()) $script`
    @test success(run(ignorestatus(command)))
else
    @test Threads.nthreads() > 1
    added_workers = addprocs(2; exeflags = `--startup-file=no --project=$(Base.active_project()) --threads=2`)
    try
        @test length(added_workers) == 2
        suite_path = joinpath(@__DIR__, "parallel_suite.jl")
        @everywhere added_workers begin
            using ForwardDiff
            using ReverseDiff
            using SciMLSensitivity
            include($suite_path)
        end
        # Stable module identity on driver and workers, also inside SafeTestsets.
        Base.include(Main, suite_path)
        @test all(pid -> remotecall_fetch(Threads.nthreads, pid) > 1, added_workers)
        println("Parallel AD runtime: Julia=", VERSION, " threads=", Threads.nthreads(),
            " workers=", added_workers, " source=", pkgdir(CorleoneBase))
        # Check where ensemble work actually runs, not just available resources.
        for mode in (EnsembleSerial(), EnsembleThreads(), EnsembleDistributed())
            locations = Main.ParallelADTests.execution_locations(mode)
            println(typeof(mode), " execution locations=", unique(locations))
            if mode isa EnsembleSerial
                @test all(location -> location[1] == getpid(), locations)
            elseif mode isa EnsembleThreads
                @test length(unique(last.(locations))) > 1
            else
                @test length(unique(first.(locations))) > 1
                @test all(location -> location[1] != getpid(), locations)
            end
        end
        Main.ParallelADTests.test_parallel_ad(AutoForwardDiff(),
            (EnsembleSerial(), EnsembleThreads(), EnsembleDistributed()))
        # ReverseDiff traces the inner solves directly, matching the sequential
        # suite's SensitivityADPassThrough configuration; compiled tapes stay
        # out of scope. safetycopy = false skips SciMLBase's default
        # per-simulation deepcopy of the template problem: with traced inputs
        # that deepcopy recurses through the ReverseDiff tape graph until the
        # stack overflows on Julia 1.10/1.11, and the harness does not need it
        # (prob_func builds fresh problems and nothing mutates the template).
        # Distributed coverage is not claimed: a GradientTape cannot survive
        # Distributed's pmap serialization (gradients come back zero and the
        # orphaned tape structures segfault the workers); see PARALLEL_RESULTS.md
        # for the reproducer and unblock path.
        Main.ParallelADTests.test_parallel_ad(AutoReverseDiff(),
            (EnsembleSerial(), EnsembleThreads());
            solve_kwargs = (; sensealg = SensitivityADPassThrough(), safetycopy = false))
    finally
        rmprocs(added_workers)
    end
end
