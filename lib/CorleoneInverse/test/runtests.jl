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
            @safetestset "Prepared experiment problems" begin
                include("core/experiment_problem.jl")
            end
            @safetestset "Experiment dynamic optimization problems" begin
                include("core/experiment_model.jl")
            end
            @safetestset "Experiment sparsity" begin
                include("core/experiment_sparsity.jl")
            end
            @safetestset "Experiment layers" begin
                include("core/experiment_layers.jl")
            end
            @safetestset "Linear regression" begin
                include("core/linear_regression.jl")
            end
            @safetestset "Priors" begin
                include("core/priors.jl")
            end
            @safetestset "Lotka Volterra" begin
                include("core/lotka_volterra.jl")
            end
            @safetestset "Multiple shooting" begin
                include("core/multiple_shooting.jl")
            end
            return @safetestset "Type-stable experiment pipeline" begin
                include("core/type_stability.jl")
            end
        end,
        qa = (; env = joinpath(@__DIR__, "qa"), body = joinpath(@__DIR__, "qa", "qa.jl")),
    )
end
