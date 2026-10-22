# Refresh existing CSV output without solving games.
# Run from lib/CorleoneGame: julia --project=. scripts/plot_library.jl [single|multiple]
using CorleoneGame

const enabled = [
    :algae,
    :antColonies,
    :avoidance,
    :biofilm,
    :ciliate,
    :fishery,
    :foraging,
    :metabolicPathway,
    :microbial,
    :mitochondria,
    :motor,
    :neural,
    :phage,
    :plantNitrogen,
    :plantWater,
    :pollination,
    :predatorPrey,
    :proteinFolding,
    :sexRatio,
    :tupelo,
    :batchFermentation,
    :circadianCycle,
    :immuneClearance,
    :seedGermination,
    :woundRepair,
]

const periodic = [
    :algae,
    :ciliate,
    :fishery,
    :metabolicPathway,
    :microbial,
    :mitochondria,
    :neural,
    :predatorPrey,
]

const modules = Dict(
    :woundRepair => :WoundRepair,
    :immuneClearance => :ImmuneClearance,
    :seedGermination => :SeedGermination,
    :batchFermentation => :BatchFermentation,
    :circadianCycle => :CircadianCycle,

    :algae => :Algae,
    :antColonies => :AntColonies,
    :avoidance => :Avoidance,
    :biofilm => :Biofilm,
    :ciliate => :Ciliate,
    :fishery => :Fishery,
    :foraging => :Foraging,
    :microbial => :Microbial,
    :mitochondria => :Mitochondria,
    :motor => :Motor,
    :metabolicPathway => :MetabolicPathway,
    :plantNitrogen => :PlantNitrogen,
    :plantWater => :PlantWater,
    :pollination => :Pollination,
    :predatorPrey => :PredatorPrey,
    :neural => :Neural,
    :phage => :Phage,
    :proteinFolding => :ProteinFolding,
    :sexRatio => :SexRatio,
    :tupelo => :Tupelo,
)

mode = isempty(ARGS) ? "single" : only(ARGS)
mode in ("single", "multiple") || error("Usage: plot_library.jl [single|multiple]")

for name in enabled
    file = joinpath(@__DIR__, "..", "library", String(name), "$(name).jl")
    include(file)
    mod = getfield(Main, modules[name])
    folder = joinpath(dirname(file), mode == "single" ? "output" : "output_multiple")
    CorleoneGame.replot_outputs(mod.game(); folder)
    GC.gc(true)
end
