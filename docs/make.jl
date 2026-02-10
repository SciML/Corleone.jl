using Pkg

Pkg.develop(path=joinpath(@__DIR__, ".."))

using Documenter, Corleone
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
    sitename="Corleone.jl",
    authors="Carl Julius Martensen, Christoph Plate, et al.",
    modules=[Corleone],
    format= Documenter.HTML(
        assets = ["assets/favicon.ico"],
        canonical = "https://docs.sciml.ai/Corleone/stable/",
        size_threshold = 1_000_000,  # bytes
    ),
    #DocumenterVitepress.MarkdownVitepress(
    #    repo = "github.com/SciML/Corleone.jl",
    #    devbranch = "main", # or master, trunk, ...
    #    devurl = "dev",
    # if you use something else than yourname.github.io/YourPackage.jl
    #),
    pages = [
        "Home" => "index.md",
        "Getting Started" => "examples/the_linear_quadratic_regulator.md",
		"Tutorials" => [
			"Linear Quadratic Regulator" => "examples/the_linear_quadratic_regulator.md",
			"Lotka Volterra Fishing" => "examples/the_lotka_volterra_fishing_problem.md",
            "Optimal Experimental Design" => "examples/the_lotka_volterra_optimal_experimental_design_problem.md",
            "Multiexperiments" => "examples/the_lotka_volterra_multiexperiment_problem.md"
		], #"tutorials.md",
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
    checkdocs=:none,
    remotes=nothing,
    plugins=[links, bib],
)

#deploydocs(
#    repo = "github.com/SciML/Corleone.jl";
#    push_preview = true
#)
#DocumenterVitepress.deploydocs(;
#    repo="github.com/SciML/Corleone.jl",
#    target=joinpath(@__DIR__, "build"),
#    branch="gh-pages",
#    devbranch="main", # or master, trunk, ...
#    push_preview=true,
#)
