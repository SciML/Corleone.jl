using Pkg

Pkg.activate(@__DIR__)
Pkg.develop([
    PackageSpec(path = joinpath(@__DIR__, "..")),
    PackageSpec(path = joinpath(@__DIR__, "..", "lib", "CorleoneOED")),
])

# Add missing direct dependencies to new and existing example environments.
# ModelingToolkit examples use their separately managed symbolic dependencies.
dependencies = [
    "OrdinaryDiffEqTsit5", "SciMLSensitivity", "Optimization", "OptimizationLBFGSB",
    "OptimizationMOI", "Ipopt", "LuxCore", "ForwardDiff", "Random",
    "SymbolicIndexingInterface", "ComponentArrays", "CairoMakie", "CSV",
    "DataFrames", "UnPack",
]
missing_dependencies = setdiff(dependencies, keys(Pkg.project().dependencies))
isempty(missing_dependencies) || Pkg.add(missing_dependencies)
Pkg.instantiate()
