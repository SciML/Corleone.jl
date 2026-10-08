# Focused check of shared-manual pages, not a separate sublibrary manual.
# Run from the repository root:
# JULIA_PKG_PRECOMPILE_AUTO=0 julia --startup-file=no --project=lib/CorleoneBase/test/docs -e 'using Pkg; Pkg.develop(path="lib/CorleoneBase"); Pkg.instantiate(); include("docs/check_corleonebase.jl")'
using Documenter, Literate, CorleoneBase, Test

const repository = dirname(@__DIR__)
@test realpath(pkgdir(CorleoneBase)) == realpath(joinpath(repository, "lib", "CorleoneBase"))
const output = get(ENV, "CORLEONEBASE_DOCS_OUTPUT", mktempdir())
mkpath(joinpath(output, "src", "examples"))
for page in ("corleonebase.md", "corleonebase_api.md")
    cp(joinpath(@__DIR__, "src", page), joinpath(output, "src", page); force = true)
end
write(joinpath(output, "src", "index.md"), """
# CorleoneBase shared-manual preview

This focused build renders the repository's shared CorleoneBase documentation:
[sequential workflow](@ref corleonebase),
[manual single shooting](@ref base_fishing), and [API](@ref base_api).
""")
Literate.markdown(
    joinpath(repository, "lib", "CorleoneBase", "examples", "lotka_fishing", "main.jl"),
    joinpath(output, "src", "examples");
    name = "manual_single_shooting_with_corleonebase", execute = true,
    edit_commit = "main",
    flavor = Literate.DocumenterFlavor()
)
makedocs(;
    root = output, sitename = "CorleoneBase in the shared manual",
    modules = [CorleoneBase], doctest = true, checkdocs = :exports,
    remotes = nothing, format = Documenter.HTML(
        edit_link = nothing, repolink = "https://github.com/SciML/Corleone.jl",
        inventory_version = string(Base.pkgversion(CorleoneBase))
    ),
    pages = [
        "Overview" => "index.md",
        "Sequential problems" => "corleonebase.md",
        "Manual single shooting" => "examples/manual_single_shooting_with_corleonebase.md",
        "API" => "corleonebase_api.md",
    ]
)
@test isfile(joinpath(output, "build", "index.html"))
@test isfile(joinpath(output, "build", "corleonebase", "index.html"))
@test isfile(joinpath(output, "build", "corleonebase_api", "index.html"))
@test isfile(joinpath(output, "build", "examples", "manual_single_shooting_with_corleonebase", "index.html"))
@info "Strict shared-page render, doctests, and executed Literate tutorial passed" output
