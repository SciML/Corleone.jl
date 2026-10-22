# Dual console/run.log output of command-line runs.
using Test, Logging
using CorleoneGame

@testset "Console and run.log receive the same output" begin
    mktempdir() do directory
        function fake_main(; shooting = :single, synthetic = false, folder)
            println("Julia stdout: $shooting / $synthetic")
            println(stderr, "Julia stderr")
            @warn "Logged warning"
            ccall(:puts, Cint, (Cstring,), "Native stdout")
            message = "Native stderr\n"
            ccall(:write, Int, (Cint, Cstring, Csize_t), 2, message, sizeof(message))
            # Check that output arrives before the run returns, not only on exit.
            flush(stdout); flush(stderr); Base.Libc.flush_cstdio()
            logfile = joinpath(folder, "run.log")
            deadline = time() + 5
            while time() < deadline && !(isfile(logfile) && occursin("Native stderr", read(logfile, String)))
                sleep(0.01)
            end
            @test occursin("Native stderr", read(logfile, String))
            println(repeat("buffer test ", 10000))
            return synthetic ? "synthetic result" : (; converged = true, shooting)
        end
        for args in (String[], ["multiple"], ["synthetic"])
            console = joinpath(directory, "console.txt")
            result = open(console, "w") do out
                redirect_stdout(out) do
                    run_logged(fake_main, directory; args)
                end
            end
            folder = joinpath(directory, args == ["multiple"] ? "output_multiple" : "output")
            log = read(joinpath(folder, "run.log"), String)
            @test read(console, String) == log
            @test all(
                occursin(marker, log) for marker in
                    ("Julia stdout", "Julia stderr", "Logged warning", "Native stdout", "Native stderr", "Finished")
            )
            @test count("Running ", log) == 1 # previous log is replaced
            @test args == ["synthetic"] ? result == "synthetic result" : result.converged
        end
        oldout, olderr, oldlogger = stdout, stderr, current_logger()
        console = joinpath(directory, "failure_console.txt")
        open(console, "w") do out
            redirect_stdout(out) do
                function failing_main(; kwargs...)
                    println("before failure")
                    error("deliberate logging failure")
                end
                @test_throws ErrorException run_logged(failing_main, directory; args = String[])
            end
        end
        log = read(joinpath(directory, "output", "run.log"), String)
        @test occursin("deliberate logging failure", log)
        @test read(console, String) == log
        @test stdout === oldout && stderr === olderr && current_logger() === oldlogger
    end
end
