using Pkg;
# Activate
Pkg.activate(@__DIR__)

if !isfile(joinpath(@__DIR__, "Project.toml"))
    # We add Corleone
    Pkg.develop(path = string(@__DIR__, "/.."))
    Pkg.develop(path = string(@__DIR__, "/../lib/CorleoneOED"))
    # We add all other deps
    Pkg.add("OrdinaryDiffEq")
    Pkg.add("SciMLSensitivity")
    Pkg.add("Optimization")
    Pkg.add("OptimizationMOI")
    Pkg.add("Ipopt")
    Pkg.add("LuxCore")
    Pkg.add("ComponentArrays")
    Pkg.add("CairoMakie")
    Pkg.add("CSV")
    Pkg.add("DataFrames")
    Pkg.add("UnPack")
else
    # We add Corleone
    Pkg.rm("Corleone")
    Pkg.add(url="https://kosinus.math.uni-magdeburg.de/mathopt/software/corleone", rev="main")
    Pkg.resolve()
end
