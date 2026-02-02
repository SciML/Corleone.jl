using Pkg

Pkg.develop(path = joinpath(@__DIR__, ".."))

using Documenter, Corleone
using DocumenterVitepress
using DocumenterInterLinks
using DocumenterCitations
using Literate
using Dates
using YAML, JSON


# Include the interlinks
include("interlinks.jl")
# Process the tutorials 
include("process_tutorials.jl")
# Include the bibliography 
bib = CitationBibliography(joinpath(@__DIR__, "src", "assets", "bibliography.bib"))

makedocs(
    sitename = "Corleone.jl",
    authors = "Carl Julius Martensen, Christoph Plate, et al.",
    modules = [Corleone],
    format = DocumenterVitepress.MarkdownVitepress(
        repo = "github.com/SciML/Corleone.jl",
        devbranch = "main", # or master, trunk, ...
        devurl = "dev",
        # if you use something else than yourname.github.io/YourPackage.jl
    ),
    pages = [
        "Home" => "index.md",
        "Getting Started" => "examples/the_linear_quadratic_regulator.md",
        "Tutorials" => "tutorials.md",
        "References" => "references.md", 
        "API" => "api.md"
        #tutorials,
        #"Examples" => [#"Optimal Control" => "./examples/lotka.md",
        #"Multiple Shooting" => "./examples/multiple_shooting.md",
        # "Optimal Experimental Design" => "./examples/lotka_oed.md",
        #"Multiexperiments" => "./examples/multiexperiments.md"
        #],
        # "API Reference" => "api.md",
    ],
    checkdocs = :none,
    remotes = nothing,
    plugins = [links, bib],
)

DocumenterVitepress.deploydocs(;
    repo = "github.com/SciML/Corleone.jl",
    target = joinpath(@__DIR__, "build"),
    branch = "gh-pages",
    devbranch = "main", # or master, trunk, ...
    push_preview = true,
)