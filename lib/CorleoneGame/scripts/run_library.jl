# Solve the library cases and write their output/ folders.
# Run from lib/CorleoneGame: julia --project=. scripts/run_library.jl [single|multiple|synthetic]
using CorleoneGame

const all = [
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


const enabled = [
    #    :algae,
    #    :antColonies,
    #    :avoidance,
    #    :biofilm,
    #    :ciliate,
    #    :fishery,
    #    :foraging,
    #    :metabolicPathway,
    #    :microbial,
    #    :mitochondria,
    #    :motor,
    #    :neural,
    #    :phage,
    #    :plantNitrogen,
    #    :plantWater,
    #    :pollination,
    #    :predatorPrey,
    #    :proteinFolding,
    #    :sexRatio,
    :tupelo,
    #    :batchFermentation,
    #    :circadianCycle,
    #    :immuneClearance,
    #    :seedGermination,
    #    :woundRepair,
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

# Replace `all` by `enabled` to solve only the selected cases.
for name in all
    file = joinpath(@__DIR__, "..", "library", String(name), "$(name).jl")
    include(file)
    mod = getfield(Main, modules[name])
    CorleoneGame.run_logged(mod.main, dirname(file); args = ARGS)
    GC.gc(true)
end
