using CorleoneGame
using SafeTestsets

# Centralized sublibrary CI sets CORLEONE_TEST_GROUP to the bare package name
# (-> "Core") or "<pkg>_<grp>" (-> "<grp>"). Fall back to GROUP, then "All", so
# local `Pkg.test()` runs (which set neither) run everything.
#   Core:    solver API on small analytic games (src/CorleoneGame.jl)
#   Utils:   plotting, CSV output and replotting, synthetic data, logging (src/utils.jl)
#   Library: every case in library/ constructs a valid game (no solves)
const _G = get(ENV, "CORLEONE_TEST_GROUP", get(ENV, "GROUP", "All"))
const _SUB = "CorleoneGame"
const GROUP = _G == _SUB ? "Core" : (startswith(_G, _SUB * "_") ? _G[(length(_SUB) + 2):end] : _G)

if GROUP in ("All", "Core")
    @safetestset "Game API" begin
        include("core/game_api.jl")
    end
    @safetestset "Options, solvers, and warm starts" begin
        include("core/options.jl")
    end
    @safetestset "Duration parameter" begin
        include("core/duration_parameter.jl")
    end
end

if GROUP in ("All", "Utils")
    @safetestset "Plots and display labels" begin
        include("utils/plots.jl")
    end
    @safetestset "Replot saved outputs" begin
        include("utils/replot_outputs.jl")
    end
    @safetestset "Synthetic observations" begin
        include("utils/synthetic_data.jl")
    end
    @safetestset "Run logging" begin
        include("utils/run_logged.jl")
    end
end

if GROUP in ("All", "Library")
    @safetestset "Library definitions" begin
        include("library/definitions.jl")
    end
end
