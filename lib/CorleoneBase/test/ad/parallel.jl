using Distributed
using SciMLBase: EnsembleSerial, EnsembleThreads, EnsembleDistributed
using DifferentiationInterface: AutoForwardDiff
using ForwardDiff
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
    finally
        rmprocs(added_workers)
    end
end
