using CorleoneOED, Documenter, Literate
using Base64: base64decode

realpath(pkgdir(CorleoneOED)) == realpath(joinpath(@__DIR__, "..")) &&
    realpath(pkgdir(Corleone)) == realpath(joinpath(@__DIR__, "..", "..", "..")) ||
    error("Develop both checked-out packages in the documentation environment; see docs/README.md.")

# Follow the root manual's convention: external PNGs keep figures out of search
# indices and allow ordinary Markdown alternative text instead of embedded HTML.
function tutorial_figures(content, output, name, alt)
    index = 0
    content = replace(
        content, r"```@raw html\n<img [^\n]+>\n```" => block -> begin
            image = match(r"src=\"data:image/png;base64,\s*([^\"]+)\"", block)
            isnothing(image) && return block
            index += 1
            filename = "$name-figure-$index.png"
            write(joinpath(output, filename), base64decode(image.captures[1]))
            "![$alt]($filename)"
        end
    )
    return replace(content, "![](" => "![$alt](")
end

# These files are generated from the executable sources, never edited by hand.
output = joinpath(@__DIR__, "src", "tutorials")
isdir(output) && rm(output; recursive = true)
for (name, alt) in (
        ("lotka_oed", "Population trajectories, fishing control, and continuous measurement densities"),
        ("lotka_multiexperiments", "Population trajectories and independent fishing and measurement schedules for one experiment"),
        ("compartmental_oed", "Relaxed measurement weights at the compartmental model's candidate sampling times"),
    )
    Literate.markdown(
        joinpath(@__DIR__, "..", "examples", name, "main.jl"), output;
        name, execute = true, flavor = Literate.DocumenterFlavor(),
        image_formats = [(MIME("image/png"), ".png")],
        postprocess = content -> tutorial_figures(content, output, name, alt),
    )
end

makedocs(;
    root = @__DIR__,
    sitename = "CorleoneOED",
    authors = "Christoph Plate, Carl Julius Martensen, Sebastian Sager, and contributors",
    modules = [CorleoneOED],
    format = Documenter.HTML(edit_link = "main"),
    pages = [
        "Overview and quick start" => "index.md",
        "Tutorials" => [
            "Continuous Lotka–Volterra design" => "tutorials/lotka_oed.md",
            "Multiple experiments" => "tutorials/lotka_multiexperiments.md",
            "Discrete compartmental design" => "tutorials/compartmental_oed.md",
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
