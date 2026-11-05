using Documenter, Corleone
import CorleoneOED

realpath(pkgdir(Corleone)) == realpath(joinpath(@__DIR__, "..")) &&
    realpath(pkgdir(CorleoneOED)) == realpath(joinpath(@__DIR__, "..", "lib", "CorleoneOED")) ||
    error("Develop this checkout in the documentation environment before building; see docs/README.md.")

using DocumenterCitations
using DocumenterInterLinks
using DocumenterLandingPage
using MaterialDocs
using Literate
using DemoCards
using Dates
using YAML

include("material_citations.jl")
include("material_images.jl")
include("draft.jl")
draft = documentation_draft()
@info "Documentation build" draft
include("process_tutorials.jl")
include("interlinks.jl")
bib = CitationBibliography(joinpath(@__DIR__, "src", "assets", "bibliography.bib"))

makedocs(
    draft = draft,
    root = @__DIR__,
    sitename = "Corleone.jl",
    authors = "Carl Julius Martensen, Christoph Plate, et al.",
    modules = [Corleone, CorleoneOED],
    format = Material3(
        theme = :ocean_depth,
        dark_mode = :toggle,
        edit_link = "main",
        repolink = "https://github.com/SciML/Corleone.jl",
        canonical = "https://docs.sciml.ai/Corleone/stable/",
        size_threshold = 1_500_000,
        assets = isnothing(demo_assets) ? String[] : [demo_assets],
    ),
    doctest = true,
    checkdocs = :exports,
    linkcheck = get(ENV, "DOCS_LINKCHECK", "false") == "true",
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "Tutorials" => [
            "Overview" => "gallery/index.md",
            "Optimal Control" => [
                "Linear Quadratic Regulator" => "examples/the_linear_quadratic_regulator.md",
                "Lotka Volterra Fishing" => "examples/the_lotka_volterra_fishing_problem.md",
                "ModelingToolkit Integration" => "examples/modelingtoolkit_integration.md",
                "LTI Control with Unknown Coefficients" => "examples/lti_control_with_unknown_coefficients.md",
            ],
            "Parameter Estimation" => [
                "Lotka–Volterra Universal Differential Equation" => "examples/lotka_volterra_universal_differential_equation.md",
            ],
            "Optimal Experimental Design" => [
                "Continuous Lotka–Volterra Design" => "examples/the_lotka_volterra_optimal_experimental_design_problem.md",
                "Discrete Compartmental Design" => "examples/compartmental_oed_problem.md",
                "Multiple Experiments" => "examples/the_lotka_volterra_multiexperiment_problem.md",
            ],
        ],
        "Manual" => [
            "Constructing Shooting Problems" => "shooting.md",
            "Scheduling Parameters" => "scheduling.md",
            "Initializing Models" => "initialization.md",
            "Inspecting Trajectories" => "trajectories.md",
            "Adding Objectives and Constraints" => "optimization_contracts.md",
            "ModelingToolkit" => "modelingtoolkit.md",
            "Selecting Differentiation Methods" => "automatic_differentiation.md",
            "Preparing Sparse Derivatives" => "sparse_derivatives.md",
            "Running Parallel Shooting" => "parallel_shooting.md",
            "Querying Symbolic Metadata" => "symbolic_metadata.md",
        ],
        "Sublibraries" => [
            "Overview" => "sublibraries.md",
            "CorleoneOED" => "oed/index.md",
            "Measurements and Information" => "oed/formulation.md",
            "Constructing and Optimizing Designs" => "oed/designs.md",
            "Reduction, AD, and Numerical Scope" => "oed/numerics.md",
            "API Reference" => "oed/api.md",
            "References and Attribution" => "oed/references.md",
        ],
        "References" => "references.md",
        "API Reference" => "api.md",
        hide("Tutorials" => "tutorials.md"),
        hide("Historical Validation" => "automatic_differentiation_history.md"),
        hide("Deprecated Constructors" => "legacy_constructors.md"),
    ],
    remotes = nothing,
    plugins = [bib, links, LandingPage()],
)

# Local documentation builds never publish the site.
if get(ENV, "CI", "false") == "true"
    deploydocs(
        repo = "github.com/SciML/Corleone.jl";
        devbranch = "main",
        push_preview = true,
    )
end
