using CorleoneOED, Documenter, DemoCards, PlutoStaticHTML
include(joinpath(@__DIR__, "..", "..", "..", "docs", "mathengine.jl"))

# Use the same tutorial card layout in the standalone and combined manuals.
tutorial_style = joinpath(@__DIR__, "src", "assets", "tutorials.css")
mkpath(dirname(tutorial_style))
cp(
    joinpath(@__DIR__, "..", "..", "..", "docs", "src", "assets", "tutorials.css"),
    tutorial_style; force = true
)

realpath(pkgdir(CorleoneOED)) == realpath(joinpath(@__DIR__, "..")) &&
    realpath(pkgdir(Corleone)) == realpath(joinpath(@__DIR__, "..", "..", "..")) ||
    error("Develop both checked-out packages in the documentation environment; see docs/README.md.")

tutorials, tutorials_cleanup, tutorial_assets = makedemos(
    "oed_examples"; root = @__DIR__, edit_branch = "main", throw_error = true
)

makedocs(;
    root = @__DIR__,
    sitename = "CorleoneOED",
    authors = "Christoph Plate, Carl Julius Martensen, Sebastian Sager, and contributors",
    modules = [CorleoneOED],
    format = Documenter.HTML(
        mathengine = notebook_mathengine(),
        edit_link = "main",
        size_threshold = 1_500_000,
        assets = isnothing(tutorial_assets) ? ["assets/tutorials.css"] : [tutorial_assets, "assets/tutorials.css"],
    ),
    pages = [
        "Overview and quick start" => "index.md",
        "Tutorials" => [
            "Choose a measurement design" => tutorials,
            "Continuous measurement schedules" => "oed_examples/experimental_design/lotka_oed.md",
            "Discrete sampling times" => "oed_examples/experimental_design/compartmental_oed.md",
            "Combine experiments" => "oed_examples/experimental_design/lotka_multiexperiments.md",
        ],
        "User guide" => [
            "Measurements and information" => "formulation.md",
            "Constructing and optimizing designs" => "designs.md",
            "Reduction, AD, and numerical scope" => "numerics.md",
        ],
        "API reference" => "api.md",
        "References and attribution" => "references.md",
    ],
    doctest = true,
    checkdocs = :exports,
    linkcheck = get(ENV, "DOCS_LINKCHECK", "false") == "true",
    warnonly = false,
)

tutorials_cleanup()
