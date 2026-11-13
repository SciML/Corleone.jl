using Documenter, Corleone
import CorleoneOED
import CorleoneInverse

realpath(pkgdir(Corleone)) == realpath(joinpath(@__DIR__, "..")) &&
    realpath(pkgdir(CorleoneOED)) == realpath(joinpath(@__DIR__, "..", "lib", "CorleoneOED")) &&
    realpath(pkgdir(CorleoneInverse)) == realpath(joinpath(@__DIR__, "..", "lib", "CorleoneInverse")) ||
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
    modules = [Corleone, CorleoneOED, CorleoneInverse],
    # Tutorial HTML exceeds Documenter's default 200 KiB cap (two-stage reactor
    # example embeds base64 plots); keep the plain HTML format and raise the cap.
    format = Documenter.HTML(size_threshold = 1_500_000, assets = ["assets/tutorial-gallery.css"]),
    #format = Material3(
    #    theme = :ocean_depth,
    #    dark_mode = :toggle,
    #    edit_link = "main",
    #    repolink = "https://github.com/SciML/Corleone.jl",
    #    canonical = "https://docs.sciml.ai/Corleone/stable/",
    #    size_threshold = 1_500_000,
    #    assets = isnothing(demo_assets) ? String[] : [demo_assets],
    #),
    doctest = true,
    checkdocs = :exports,
    linkcheck = get(ENV, "DOCS_LINKCHECK", "false") == "true",
    pages = [
        "Home" => "index.md",
        "Getting started" => "getting_started.md",
        "Tutorials" => [
            "Overview" => "gallery/index.md",
            "Optimal control" => [
                "Linear quadratic regulator" => "examples/the_linear_quadratic_regulator.md",
                "Lotka–Volterra fishing problem" => "examples/the_lotka_volterra_fishing_problem.md",
                "Path and terminal constraints" => "examples/lotka_volterra_path_constraints.md",
                "Minimal-time problem" => "examples/minimal_time_lotka_volterra.md",
                "Fishing as an index-1 DAE" => "examples/lotka_volterra_fishing_dae.md",
                "ModelingToolkit integration" => "examples/modelingtoolkit_integration.md",
                "LTI control with unknown coefficients" => "examples/lti_control_with_unknown_coefficients.md",
                "Two-stage chemical reactor" => "examples/optimal_control_of_a_two_stage_reactor_system.md",
                "Multistage Lotka–Volterra" => "examples/multistage_lotka_volterra.md",
                "Switching-time optimization" => "examples/lotka_volterra_switching_times.md",
            ],
            "Parameter estimation" => [
                "Lotka–Volterra parameter estimation" => "examples/inverse_lotka_plain.md",
                "Joint Lotka–Volterra experiments" => "examples/inverse_lotka_multiexperiment.md",
                "Controls from experimental design" => "examples/inverse_lotka_oed_control.md",
                "Multiple-shooting parameter estimation" => "examples/inverse_multiple_shooting.md",
                "SIR estimation across two regions" => "examples/inverse_sir_multicity.md",
                "Robertson DAE parameter estimation" => "examples/inverse_rober_dae.md",
                "Learning Lotka–Volterra interactions" => "examples/lotka_volterra_universal_differential_equation.md",
                "Neural reaction term with priors" => "examples/inverse_ude_growth.md",
                "EGF signaling across doses" => "examples/inverse_egf_signaling.md",
            ],
            "Optimal experimental design" => [
                "Lotka–Volterra experimental design" => "examples/the_lotka_volterra_optimal_experimental_design_problem.md",
                "Discrete compartmental design" => "examples/compartmental_oed_problem.md",
                "Multiple experiments" => "examples/the_lotka_volterra_multiexperiment_problem.md",
            ],
        ],
        "Manual" => [
            "Scheduling parameters" => "scheduling.md",
            "Single and multiple shooting" => "shooting.md",
            "Initial guesses" => "initialization.md",
            "Trajectories" => "trajectories.md",
            "Objectives and constraints" => "optimization_contracts.md",
            "Multi-stage problems" => "multistage.md",
            "ModelingToolkit" => "modelingtoolkit.md",
            "Automatic differentiation" => "automatic_differentiation.md",
            "Sparse derivatives" => "sparse_derivatives.md",
            "Parallel shooting" => "parallel_shooting.md",
            "Symbolic metadata" => "symbolic_metadata.md",
        ],
        "Sublibraries" => [
            "Overview" => "sublibraries.md",
            "CorleoneInverse" => [
                "Parameter estimation" => "inverse/index.md",
                "Observations and parameters" => "inverse/observations.md",
                "Multiple experiments" => "inverse/experiments.md",
                "API reference" => "inverse/api.md",
            ],
            "CorleoneOED" => [
                "Overview" => "oed/index.md",
                "Measurements and Fisher information" => "oed/formulation.md",
                "Constructing and optimizing designs" => "oed/designs.md",
                "Reduction, AD, and numerical scope" => "oed/numerics.md",
                "API reference" => "oed/api.md",
                "References and attribution" => "oed/references.md",
            ],
        ],
        "References" => "references.md",
        "API reference" => "api.md",
        hide("Tutorials" => "tutorials.md"),
        hide("Historical validation" => "automatic_differentiation_history.md"),
        hide("Deprecated constructors" => "legacy_constructors.md"),
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
