using Pkg

# Document the checked-out root package, independently of the sibling libraries.
Pkg.develop(PackageSpec(path = joinpath(@__DIR__, "..")))

using Documenter, Corleone
using DocumenterCitations
using Literate
using Dates
using YAML

include("process_tutorials.jl")
bib = CitationBibliography(joinpath(@__DIR__, "src", "assets", "bibliography.bib"))

makedocs(
    root = @__DIR__,
    sitename = "Corleone.jl",
    authors = "Carl Julius Martensen, Christoph Plate, et al.",
    modules = [Corleone],
    format = Documenter.HTML(
        edit_link = "main",
        repolink = "https://github.com/SciML/Corleone.jl",
        canonical = "https://docs.sciml.ai/Corleone/stable/",
        size_threshold = 1_500_000,
    ),
    doctest = true,
    checkdocs = :exports,
    linkcheck = get(ENV, "DOCS_LINKCHECK", "false") == "true",
    pages = [
        "Home" => "index.md",
        "Tutorials" => [
            "Overview" => "tutorials.md",
            "Linear Quadratic Regulator" => "examples/the_linear_quadratic_regulator.md",
            "Lotka Volterra Fishing" => "examples/the_lotka_volterra_fishing_problem.md",
            "ModelingToolkit Integration" => "examples/modelingtoolkit_integration.md",
            "LTI Control with Unknown Coefficients" => "examples/lti_control_with_unknown_coefficients.md",
        ],
        "Initialization" => "initialization.md",
        "Optimization Contracts" => "optimization_contracts.md",
        "ModelingToolkit" => "modelingtoolkit.md",
        "Automatic Differentiation" => "automatic_differentiation.md",
        "Parallel Shooting" => "parallel_shooting.md",
        "Symbolic Bookkeeping" => "symbolic_metadata.md",
        "References" => "references.md",
        "API" => "api.md",
        "Deprecated Constructors" => "legacy_constructors.md",
    ],
    remotes = nothing,
    plugins = [bib],
)

# Local documentation builds never publish the site.
if get(ENV, "CI", "false") == "true"
    deploydocs(
        repo = "github.com/SciML/Corleone.jl";
        devbranch = "main",
        push_preview = true,
    )
end
