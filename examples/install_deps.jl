using Pkg

repository = normpath(joinpath(@__DIR__, ".."))
for (project, packages) in (
        ("docs", (".", "lib/CorleoneOED", "lib/OptimalControlBenchmarks")),
        ("lib/CorleoneOED/docs", (".", "lib/CorleoneOED")),
    )
    Pkg.activate(joinpath(repository, project))
    Pkg.develop([PackageSpec(path = joinpath(repository, package)) for package in packages])
    Pkg.instantiate()
end
