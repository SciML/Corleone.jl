using CorleoneParameterEstimation
using SafeTestsets
using SciMLTesting

# Centralized sublibrary CI sets CORLEONE_TEST_GROUP to the bare package name
# (-> "Core") or "<pkg>_<grp>" (-> "<grp>"). Fall back to GROUP, then "All", so
# local `Pkg.test()` runs (which set neither) run everything. Map the value to
# the standard Core/QA section names this file dispatches on.
const _G = get(ENV, "CORLEONE_TEST_GROUP", get(ENV, "GROUP", "All"))
const _SUB = "CorleoneParameterEstimation"
const GROUP = _G == _SUB ? "Core" : (startswith(_G, _SUB * "_") ? _G[(length(_SUB) + 2):end] : _G)

withenv("GROUP" => GROUP) do
    run_tests(;
        core = function ()
            @safetestset "Exponential decay" begin
                include("core/exp_decay.jl")
            end
            @safetestset "Multi-experiment" begin
                include("core/multi_experiment.jl")
            end
            @safetestset "Fixed controls" begin
                include("core/fixed_controls.jl")
            end
            #@safetestset "Lotka Volterra" begin
            #    include("core/lotka_volterra.jl")
            #end
            #return @safetestset "Uncertainty" begin
            #    include("core/uncertainty.jl")
            #end
        end,
        qa = (; env = joinpath(@__DIR__, "qa"), body = joinpath(@__DIR__, "qa", "qa.jl")),
    )
end
