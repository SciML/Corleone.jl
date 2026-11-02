using Pkg

# Julia 1.10 ignores [sources]; load all documentation packages from this checkout.
Pkg.develop(
    [
        PackageSpec(path = joinpath(@__DIR__, "..")),
        PackageSpec(path = joinpath(@__DIR__, "..", "lib", "CorleoneOED")),
        PackageSpec(path = joinpath(@__DIR__, "..", "lib", "OptimalControlBenchmarks")),
    ]
)
Pkg.instantiate()

using Documenter, Corleone
import CorleoneOED
import OptimalControlBenchmarks

realpath(pkgdir(Corleone)) == realpath(joinpath(@__DIR__, "..")) &&
    realpath(pkgdir(CorleoneOED)) == realpath(joinpath(@__DIR__, "..", "lib", "CorleoneOED")) ||
    error("Develop this checkout in the documentation environment before building; see docs/README.md.")
realpath(pkgdir(OptimalControlBenchmarks)) ==
    realpath(joinpath(@__DIR__, "..", "lib", "OptimalControlBenchmarks")) ||
    error("Develop OptimalControlBenchmarks from this checkout; see docs/README.md.")

using DocumenterCitations
using DemoCards, PlutoStaticHTML
include("mathengine.jl")

# DemoCards uses this directory as PlutoStaticHTML's previous_dir.
pluto_output = joinpath(@__DIR__, "pluto_output")
if get(ENV, "CI", "false") == "true" && !isdir(pluto_output)
    mktempdir() do previous
        try
            run(`git clone --quiet --depth=1 --filter=blob:none --sparse --branch=gh-pages https://github.com/SciML/Corleone.jl.git $previous`)
            run(`git -C $previous sparse-checkout set dev/pluto_output`)
            cache = joinpath(previous, "dev", "pluto_output")
            isdir(cache) && cp(cache, pluto_output)
        catch err
            @warn "Could not restore the Pluto cache; notebooks will be evaluated." exception = err
        end
    end
end

include("prepare_pages.jl")
tutorials, tutorials_cleanup, tutorial_assets = makedemos(
    "examples"; root = @__DIR__, edit_branch = "main", throw_error = true
)
oed_tutorials, oed_cleanup, oed_assets = makedemos(
    "../lib/CorleoneOED/docs/oed_examples";
    root = @__DIR__, edit_branch = "main", throw_error = true
)
# Collect both sets of cards without duplicating the authoritative notebooks.
oed_page = DemoCards.DemoPage(joinpath(@__DIR__, "..", "lib", "CorleoneOED", "docs", "oed_examples"))
templates, _ = DemoCards.cardtheme("grid"; root = @__DIR__)
oed_cards = DemoCards.generate(oed_page.sections, templates)
open(joinpath(@__DIR__, "src", tutorials), "a") do io
    write(io, "\n", replace(oed_cards, "covers/" => "../oed_examples/covers/"))
end
rm(joinpath(@__DIR__, "src", oed_tutorials))
bib = CitationBibliography(joinpath(@__DIR__, "src", "assets", "bibliography.bib"))

makedocs(
    root = @__DIR__,
    sitename = "Corleone.jl",
    authors = "Carl Julius Martensen, Christoph Plate, et al.",
    modules = [Corleone, CorleoneOED, OptimalControlBenchmarks],
    format = Documenter.HTML(
        mathengine = notebook_mathengine(),
        edit_link = "main",
        repolink = "https://github.com/SciML/Corleone.jl",
        canonical = "https://docs.sciml.ai/Corleone/stable/",
        size_threshold = 1_500_000,
        assets = unique(filter(!isnothing, [tutorial_assets, oed_assets, "assets/tutorials.css"])),
    ),
    doctest = true,
    checkdocs = :exports,
    linkcheck = get(ENV, "DOCS_LINKCHECK", "false") == "true",
    pages = [
        "Home" => "index.md",
        "Tutorials" => [
            "Gallery" => tutorials,
            "Control fundamentals" => [
                "Track a target" => "examples/optimal_control/linear_quadratic.md",
                "Control predator–prey dynamics" => "examples/optimal_control/lotka_fishing.md",
            ],
            "Symbolic models and constraints" => [
                "Build a model from equations" => "examples/modelingtoolkit/lotka.md",
                "Estimate coefficients and controls" => "examples/modelingtoolkit/lti.md",
            ],
            "Measurement design" => [
                "Continuous measurement schedules" => "oed_examples/experimental_design/lotka_oed.md",
                "Discrete sampling times" => "oed_examples/experimental_design/compartmental_oed.md",
                "Combine experiments" => "oed_examples/experimental_design/lotka_multiexperiments.md",
            ],
        ],
        "User Guide" => [
            "Shooting Formulation" => "shooting.md",
            "Trajectories" => "trajectories.md",
            "Initialization" => "initialization.md",
            "Optimization Contracts" => "optimization_contracts.md",
            "ModelingToolkit" => "modelingtoolkit.md",
            "Parallel Shooting" => "parallel_shooting.md",
            "Symbolic Bookkeeping" => "symbolic_metadata.md",
        ],
        "Optimal Experimental Design" => [
            "Overview" => "oed/index.md",
            "Measurements and Information" => "oed/formulation.md",
            "Constructing and Optimizing Designs" => "oed/designs.md",
            "Reduction, AD, and Numerical Scope" => "oed/numerics.md",
            "API Reference" => "oed/api.md",
            "References and Attribution" => "oed/references.md",
        ],
        "Optimal Control Benchmarks" => "benchmarks/index.md",
        "Automatic Differentiation" => [
            "Backend Guide" => "automatic_differentiation.md",
            "Historical Validation" => "automatic_differentiation_history.md",
        ],
        "References" => "references.md",
        "API" => "api.md",
        "Deprecated Constructors" => "legacy_constructors.md",
    ],
    remotes = nothing,
    plugins = [bib],
)

tutorials_cleanup()
oed_cleanup()

# Preserve the raw Markdown for the next build, as documented by PlutoStaticHTML.
cp(pluto_output, joinpath(@__DIR__, "build", "pluto_output"); force = true)

# Local documentation builds never publish the site.
if get(ENV, "CI", "false") == "true"
    deploydocs(
        repo = "github.com/SciML/Corleone.jl";
        devbranch = "main",
        push_preview = true,
    )
end
