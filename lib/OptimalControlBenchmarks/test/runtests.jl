using OptimalControlBenchmarks
using Test
using SafeTestsets
using SciMLTesting

# Centralized sublibrary CI sets CORLEONE_TEST_GROUP to the bare package name
# (-> "Core") or "<pkg>_<grp>" (-> "<grp>"). Fall back to GROUP, then "All", so
# local `Pkg.test()` runs (which set neither) run everything. Map the value to
# the standard Core/QA section names this file dispatches on.
const _G = get(ENV, "CORLEONE_TEST_GROUP", get(ENV, "GROUP", "All"))
const _SUB = "OptimalControlBenchmarks"
const GROUP = _G == _SUB ? "Core" : (startswith(_G, _SUB * "_") ? _G[(length(_SUB) + 2):end] : _G)

withenv("GROUP" => GROUP) do
    run_tests(;
        core = function ()
            # No Core unit tests yet; running the benchmarks requires the full MTK +
            # Ipopt/MOI optimal-control stack and is too expensive for CI. Keep a
            # trivial assertion so the Core group still emits a Test Summary.
            return @test 1 == 1
        end,
        qa = (; env = joinpath(@__DIR__, "qa"), body = joinpath(@__DIR__, "qa", "qa.jl")),
    )
end
