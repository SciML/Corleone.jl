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
using Literate
using Dates
using YAML

include("process_tutorials.jl")
bib = CitationBibliography(joinpath(@__DIR__, "src", "assets", "bibliography.bib"))

makedocs(
    root = @__DIR__,
    sitename = "Corleone.jl",
    authors = "Carl Julius Martensen, Christoph Plate, et al.",
    modules = [Corleone, CorleoneOED, OptimalControlBenchmarks],
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
            "Tutorials" => [
                "Continuous Lotka–Volterra Design" => "examples/the_lotka_volterra_optimal_experimental_design_problem.md",
                "Discrete Compartmental Design" => "examples/compartmental_oed_problem.md",
                "Multiple Experiments" => "examples/the_lotka_volterra_multiexperiment_problem.md",
            ],
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

# Local documentation builds never publish the site.
if get(ENV, "CI", "false") == "true"
    deploydocs(
        repo = "github.com/SciML/Corleone.jl";
        devbranch = "main",
        push_preview = true,
    )
end
