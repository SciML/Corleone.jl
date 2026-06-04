using OptimalControlBenchmarks
using Test

# Centralized sublibrary CI emits GROUP as the bare package name (-> "Core") or
# "<pkg>_<grp>" (-> "<grp>"). Map it to the bare section names this file
# dispatches on. GROUP="All" keeps local `Pkg.test()` runs running everything.
const _G = get(ENV, "GROUP", "All")
const _SUB = "OptimalControlBenchmarks"
const GROUP = _G == _SUB ? "Core" : (startswith(_G, _SUB * "_") ? _G[(length(_SUB) + 2):end] : _G)

if GROUP == "All" || GROUP == "Core"
    @test 1 == 1
end
