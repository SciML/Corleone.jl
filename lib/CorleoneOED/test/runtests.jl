using CorleoneOED
using SafeTestsets
using Pkg

# QA (Aqua) runs in an isolated environment (test/qa) so its tooling deps never
# enter the main test target's resolve. On Julia < 1.11 the [sources] table is
# ignored, so develop the package and its in-repo sibling by path.
function activate_qa_env()
    Pkg.activate(joinpath(@__DIR__, "qa"))
    Pkg.develop([
        Pkg.PackageSpec(path = joinpath(@__DIR__, "..")),
        Pkg.PackageSpec(path = joinpath(@__DIR__, "..", "..", ".."))
    ])
    return Pkg.instantiate()
end

# Centralized sublibrary CI sets CORLEONE_TEST_GROUP to the bare package name
# (-> "Core") or "<pkg>_<grp>" (-> "<grp>"). Fall back to GROUP, then "All", so
# local `Pkg.test()` runs (which set neither) run everything. Map the value to
# the standard Core/QA section names this file dispatches on.
const _G = get(ENV, "CORLEONE_TEST_GROUP", get(ENV, "GROUP", "All"))
const _SUB = "CorleoneOED"
const GROUP = _G == _SUB ? "Core" : (startswith(_G, _SUB * "_") ? _G[(length(_SUB) + 2):end] : _G)

if GROUP == "All" || GROUP == "Core"
    @safetestset "1D Example" begin
        include("core/1d_oed.jl")
    end
    @safetestset "Lotka Volterra" begin
        include("core/lotka_oed.jl")
    end
    @safetestset "Lotka Volterra SVD" begin
        include("core/lotka_oed_svd.jl")
    end
end

if GROUP == "All" || GROUP == "QA"
    activate_qa_env()
    @safetestset "Code quality (Aqua.jl)" include("qa/qa.jl")
end
