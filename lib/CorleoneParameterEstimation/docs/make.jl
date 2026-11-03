using CorleoneParameterEstimation, Documenter, Literate
using Base64: base64decode

realpath(pkgdir(CorleoneParameterEstimation)) == realpath(joinpath(@__DIR__, "..")) &&
    realpath(pkgdir(Corleone)) == realpath(joinpath(@__DIR__, "..", "..", "..")) ||
    error("Develop the checked-out packages in the documentation environment; see docs/README.md.")

# Follow the root manual's convention: external images keep figures out of search
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
        ("lotka_plain", "Lotka–Volterra fit of noisy observations with parameter uncertainties"),
        ("lotka_multiexperiment", "Joint Lotka–Volterra fit over two experiments"),
        ("lotka_oed_control", "Lotka–Volterra fit under an OED-designed fixed fishing control"),
        ("multiple_shooting", "Multiple-shooting Lotka–Volterra fit and continuity residuals"),
        ("rober_dae", "Robertson DAE fit of a parameter subset with log transforms"),
        ("sir_multicity", "Joint SIR epidemic fit across two regions"),
        ("ude_growth", "Universal differential equation growth model with a Lux network"),
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
    sitename = "CorleoneParameterEstimation",
    authors = "Christoph Plate, Carl Julius Martensen, Sebastian Sager, and contributors",
    modules = [CorleoneParameterEstimation],
    format = Documenter.HTML(edit_link = "main"),
    pages = [
        "Overview and quick start" => "index.md",
        "Tutorials" => [
            "Plain Lotka–Volterra fit" => "tutorials/lotka_plain.md",
            "Multiple experiments" => "tutorials/lotka_multiexperiment.md",
            "Fixed controls from an OED design" => "tutorials/lotka_oed_control.md",
            "Multiple shooting" => "tutorials/multiple_shooting.md",
            "Robertson DAE" => "tutorials/rober_dae.md",
            "SIR across two regions" => "tutorials/sir_multicity.md",
            "Universal differential equation" => "tutorials/ude_growth.md",
        ],
        "User guide" => [
            "Problem setup and data" => "formulation.md",
            "Fitting, transforms, and regularization" => "numerics.md",
        ],
        "API reference" => "api.md",
    ],
    doctest = true,
    checkdocs = :exports,
    linkcheck = get(ENV, "DOCS_LINKCHECK", "false") == "true",
    warnonly = false,
)
