using Pkg;
# Activate
Pkg.activate(@__DIR__)

if !isfile(joinpath(@__DIR__, "Project.toml"))
    # We add Corleone
    if ("-ParentCorleone" in ARGS || "--ParentCorleone" in ARGS)
        Pkg.develop(path = joinpath(@__DIR__, ".."))
    else   
        Pkg.add(url="https://kosinus.math.uni-magdeburg.de/mathopt/software/corleone", rev="main")
    end
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
    if ("-ParentCorleone" in ARGS || "--ParentCorleone" in ARGS)
        Pkg.develop(path = joinpath(Base.@__DIR__, ".."))
    else  
        Pkg.add(url="https://kosinus.math.uni-magdeburg.de/mathopt/software/corleone", rev="main")
    end
    Pkg.resolve()
end
