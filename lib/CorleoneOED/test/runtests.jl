using CorleoneOED
using SafeTestsets

# Centralized sublibrary CI emits GROUP as the bare package name (-> "Core") or
# "<pkg>_<grp>" (-> "<grp>"). Map it to the standard Core/QA section names this
# file dispatches on. GROUP="All" keeps local `Pkg.test()` runs running everything.
const _G = get(ENV, "GROUP", "All")
const _SUB = "CorleoneOED"
const GROUP = _G == _SUB ? "Core" : (startswith(_G, _SUB * "_") ? _G[(length(_SUB) + 2):end] : _G)

if GROUP == "All" || GROUP == "Core"
    @safetestset "1D Example" begin
        include("1d_oed.jl")
    end
    @safetestset "Lotka Volterra" begin
        include("lotka_oed.jl")
    end
    @safetestset "Lotka Volterra SVD" begin
        include("lotka_oed_svd.jl")
    end
end

if GROUP == "All" || GROUP == "QA"
    @safetestset "Code quality (Aqua.jl)" begin
        using Aqua
        using CorleoneOED
        Aqua.test_all(CorleoneOED)
    end
end
