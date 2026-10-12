using Distributed
using SciMLBase: SciMLBase, EnsembleSerial, EnsembleThreads, EnsembleDistributed
using DifferentiationInterface: AutoForwardDiff, AutoReverseDiff, AutoZygote, AutoMooncake,
    AutoMooncakeForward
using ForwardDiff
using Mooncake
using ReverseDiff
using SciMLSensitivity: SciMLSensitivity, SensitivityADPassThrough
using Zygote
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
        mooncake_rules = abspath(joinpath(@__DIR__, "../../ext/CorleoneBaseMooncakeExtension.jl"))
        @everywhere added_workers begin
            using ForwardDiff
            using Mooncake
            using ReverseDiff
            using Zygote
            using SciMLSensitivity
            include($mooncake_rules)
            include($suite_path)
        end
        # Stable module identity on driver and workers, also inside SafeTestsets.
        Base.include(Main, suite_path)
        Base.include(Main, mooncake_rules)
        for (package, extension) in (
                (SciMLBase, :SciMLBaseMooncakeExt),
                (SciMLSensitivity, :SciMLSensitivityMooncakeExt),
            )
            @test Base.get_extension(package, extension) !== nothing
            @test all(
                pid -> remotecall_fetch(Base.get_extension, pid, package, extension) !== nothing,
                added_workers
            )
        end
        @test all(pid -> remotecall_fetch(Threads.nthreads, pid) > 1, added_workers)
        println(
            "Parallel AD runtime: Julia=", VERSION, " threads=", Threads.nthreads(),
            " workers=", added_workers, " source=", pkgdir(CorleoneBase)
        )
        flush(stdout)
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
        Main.ParallelADTests.test_parallel_ad(
            AutoForwardDiff(),
            (EnsembleSerial(), EnsembleThreads(), EnsembleDistributed())
        )
        flush(stdout)
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
        Main.ParallelADTests.test_parallel_ad(
            AutoReverseDiff(),
            (EnsembleSerial(), EnsembleThreads());
            solve_kwargs = (; sensealg = SensitivityADPassThrough(), safetycopy = false)
        )
        flush(stdout)
        # Zygote registers across the full ensemble matrix. It adjoins the actual
        # ensemble solve: the CorleoneBaseZygoteExtension runs the real (threaded
        # or distributed) solve as an opaque forward and re-differentiates the
        # trajectory problems serially in the pullback, so threaded wrapped
        # trajectories do not race Zygote's recorder and distributed workers do
        # not serialize a tape. Ensemble solutions are mode-independent, so the
        # adjoint has the same primal values as the forward that the
        # execution-location probes observe.
        Main.ParallelADTests.test_parallel_ad(
            AutoZygote(),
            (EnsembleSerial(), EnsembleThreads(), EnsembleDistributed())
        )
        flush(stdout)
        # Mooncake runs the real parallel primal, then differentiates the actual
        # deterministic trajectory solves serially. The hook supplies fresh
        # problems, so no template mutation or per-trajectory safetycopy is needed.
        Main.ParallelADTests.test_parallel_ad(
            AutoMooncake(),
            (EnsembleSerial(), EnsembleThreads(), EnsembleDistributed());
            solve_kwargs = (; safetycopy = false)
        )
        flush(stdout)
        # AutoMooncakeForward is Mooncake's forward mode: the shared extension's
        # frule!! runs the genuine requested-mode parallel primal, then
        # differentiates the actual deterministic trajectory solves serially with
        # the incoming Duals, so threaded and distributed execution is observed by
        # the probes and gradients carry the real per-trajectory contributions.
        # The same extension file supplies both the reverse-mode rrule and this
        # forward-mode frule; safetycopy = false is required by the rule.
        Main.ParallelADTests.test_parallel_ad(
            AutoMooncakeForward(),
            (EnsembleSerial(), EnsembleThreads(), EnsembleDistributed());
            solve_kwargs = (; safetycopy = false)
        )
        flush(stdout)
    finally
        rmprocs(added_workers)
    end
    @test all(pid -> !(pid in workers()), added_workers)
end
