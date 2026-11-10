using CorleoneInverse
using SafeTestsets
using SciMLTesting

# Centralized sublibrary CI sets CORLEONE_TEST_GROUP to the bare package name
# (-> "Core") or "<pkg>_<grp>" (-> "<grp>"). Fall back to GROUP, then "All", so
# local `Pkg.test()` runs (which set neither) run everything.
const _G = get(ENV, "CORLEONE_TEST_GROUP", get(ENV, "GROUP", "All"))
const _SUB = "CorleoneInverse"
const GROUP = _G == _SUB ? "Core" : (startswith(_G, _SUB * "_") ? _G[(length(_SUB) + 2):end] : _G)

withenv("GROUP" => GROUP) do
    run_tests(;
        core = function ()
            @safetestset "Linear regression" begin
                include("core/linear_regression.jl")
            end
            @safetestset "Priors" begin
                include("core/priors.jl")
            end
            @safetestset "Lotka Volterra" begin
                include("core/lotka_volterra.jl")
            end
            return @safetestset "Multiple shooting" begin
                include("core/multiple_shooting.jl")
            end
        end,
        qa = (; env = joinpath(@__DIR__, "qa"), body = joinpath(@__DIR__, "qa", "qa.jl")),
    )
end
