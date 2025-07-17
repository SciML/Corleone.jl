using Pkg;
# Activate
Pkg.activate(@__DIR__)

if !isfile(joinpath(@__DIR__, "Project.toml"))
    # We add Corleone
    Pkg.add(url="https://kosinus.math.uni-magdeburg.de/mathopt/software/corleone", rev="main")
    # We add all other deps 
    Pkg.add("OrdinaryDiffEqTsit5")
    Pkg.add("OrdinaryDiffEq")
    Pkg.add("SciMLSensitivity")
    Pkg.add("Optimization")
    Pkg.add("OptimizationMOI")
    Pkg.add("Ipopt")
    Pkg.add("ModelingToolkit")
    Pkg.add("CairoMakie")
else
    # We add Corleone
    Pkg.rm("Corleone")
    Pkg.add(url="https://kosinus.math.uni-magdeburg.de/mathopt/software/corleone", rev="main")
    Pkg.resolve()
end
