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
            @safetestset "Grids and registry" begin
                include("core/grids_registry.jl")
            end
            return @safetestset "Corleone driver" begin
                include("core/solver.jl")
            end
        end,
        qa = (; env = joinpath(@__DIR__, "qa"), body = joinpath(@__DIR__, "qa", "qa.jl")),
    )
end
