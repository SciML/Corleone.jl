using Pkg;
# Activate
Pkg.activate(@__DIR__)

if !isfile(joinpath(@__DIR__, "Project.toml"))
    # We add Corleone
    Pkg.add(url="https://kosinus.math.uni-magdeburg.de/mathopt/software/corleone", rev="CP/dev_cleanup")
    # We add all other deps
    Pkg.add("OrdinaryDiffEq")
    Pkg.add("SciMLSensitivity")
    Pkg.add("Optimization")
    Pkg.add("OptimizationMOI")
    Pkg.add("Ipopt")
    Pkg.add("LuxCore")
    Pkg.add("ComponentArrays")
    Pkg.add("CairoMakie")
else
    # We add Corleone
    Pkg.rm("Corleone")
    Pkg.add(url="https://kosinus.math.uni-magdeburg.de/mathopt/software/corleone", rev="CP/dev_cleanup")
    Pkg.resolve()
end
