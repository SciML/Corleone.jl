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
            @safetestset "Per-quantity observation times" begin
                include("core/per_quantity.jl")
            end
            @safetestset "Diagnostics" begin
                include("core/diagnostics.jl")
            end
            @safetestset "Profile likelihood" begin
                include("core/profile_likelihood.jl")
            end
            @safetestset "Type stability" begin
                include("core/type_stability.jl")
            end
            @safetestset "Parameter transforms" begin
                include("core/transforms.jl")
            end
            @safetestset "Regularization" begin
                include("core/regularization.jl")
            end
            return @safetestset "Multiple shooting" begin
                include("core/multiple_shooting.jl")
            end
        end,
        groups = Dict(
            "Extensions" => (;
                body = function ()
                    @safetestset "CairoMakie plotting" begin
                        include("core/plotting.jl")
                    end
                    return @safetestset "Optimization extension" begin
                        include("core/optimization_extension.jl")
                    end
                end,
            ),
            # Reverse-mode AD through the out-of-place residual needs Zygote and
            # SciMLSensitivity, kept in a dedicated environment so they never enter the
            # light main test target (which also runs on Julia pre-release).
            "AD" => (;
                env = joinpath(@__DIR__, "ad"),
                body = joinpath(@__DIR__, "ad", "reverse_mode.jl"),
            ),
        ),
        all = ["Core", "Extensions"],
        qa = (; env = joinpath(@__DIR__, "qa"), body = joinpath(@__DIR__, "qa", "qa.jl")),
    )
end
