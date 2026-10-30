using Base64: base64decode

function extract_metadata(file_path)
    content = read(file_path, String)
    # Match the block between #src --- and #src ---
    m = match(r"#src ---(?:\r?\n)((?:#src .*\r?\n)*)#src ---", content)

    if m !== nothing
        # Remove the "#src " prefix from each line to get valid YAML
        yaml_text = replace(m.captures[1], r"^#src\s?"m => "")
        return YAML.load(yaml_text)
    end
    # Try to get the first header for the title
    return nothing
end

# Keep plot data out of HTML pages and Documenter's text search index.
function tutorial_figures(content, title, outpath, name)
    index = 0
    content = replace(
        content, r"```@raw html\n<img [^\n]+>\n```" => block -> begin
            image = match(r"src=\"data:image/png;base64,\s*([^\"]+)\"", block)
            isnothing(image) && return block
            index += 1
            filename = "$name-figure-$index.png"
            write(joinpath(outpath, filename), base64decode(image.captures[1]))
            "![Trajectory plot for $title]($filename)"
        end
    )
    return replace(content, "![](" => "![Trajectory plot for $title](")
end

function preprocess_script(content)
    content = replace(
        content, "TODAY" => string(Date(now())),
        "[`OptimizationLBFGSB.LBFGSB`](@extref)" =>
            "[`OptimizationLBFGSB.LBFGSB`](https://docs.sciml.ai/Optimization/stable/optimization_packages/lbfgsb/)"
    )
    content *= """
            # ## Appendix
            # ```@raw html
            # <details><summary>This example was built using these direct dependencies,</summary>
            # ```
            using Pkg
            Pkg.status()
            # ```@raw html
            # </details>
            # ```
            # ```@raw html
            # <details><summary>and using this machine and Julia version.</summary>
            # ```
            using InteractiveUtils
            InteractiveUtils.versioninfo()
            # ```@raw html
            # </details>
            # ```
            # ```@raw html
            # <details><summary>A more complete overview of all dependencies and their versions is also provided.</summary>
            # ```
            using Pkg
            Pkg.status(; mode = PKGMODE_MANIFEST)
            # ```@raw html
            # </details>
            # ```
    """
    return content
end


function make_tutorial(path; name = nothing)
    isfile(path) || return nothing
    metadata = extract_metadata(path)
    _..., scriptname = splitpath(path)
    title = get(something(metadata, Dict()), "title", first(splitext(scriptname)))
    fname = isnothing(name) ? replace(lowercase(title), " " => "_") : name
    outpath = joinpath(@__DIR__, "src", "examples")
    isdir(outpath) || mkdir(outpath)
    Literate.markdown(
        path, outpath,
        execute = true,
        preprocess = preprocess_script,
        postprocess = content -> tutorial_figures(content, title, outpath, fname),
        flavor = Literate.DocumenterFlavor(),
        name = fname
    )
    metadata = something(metadata, Dict("title" => fname))
    metadata["link"] = joinpath(".", "examples", fname)
    return metadata
end

# Both sites execute the same authoritative OED examples. Explicit names preserve
# the OED tutorial URLs already published by main, independently of source titles.
example_output = joinpath(@__DIR__, "src", "examples")
isdir(example_output) && rm(example_output; recursive = true)

tutorials = map(
    [
        (joinpath(@__DIR__, "..", "examples", "linear_quadratic", "main.jl"), nothing),
        (joinpath(@__DIR__, "..", "examples", "lotka_fishing_optimal_control", "main.jl"), nothing),
        (joinpath(@__DIR__, "..", "examples", "mtk_lotka", "main.jl"), nothing),
        (joinpath(@__DIR__, "..", "examples", "mtk_lti", "main.jl"), nothing),
        (
            joinpath(@__DIR__, "..", "lib", "CorleoneOED", "examples", "lotka_oed", "main.jl"),
            "the_lotka_volterra_optimal_experimental_design_problem",
        ),
        (
            joinpath(@__DIR__, "..", "lib", "CorleoneOED", "examples", "lotka_multiexperiments", "main.jl"),
            "the_lotka_volterra_multiexperiment_problem",
        ),
        (
            joinpath(@__DIR__, "..", "lib", "CorleoneOED", "examples", "compartmental_oed", "main.jl"),
            "compartmental_oed_problem",
        ),
    ]
) do (path, name)
    make_tutorial(path; name)
end

# Keep the OED guide/API authoritative in the sublibrary; generate root-site pages
# with local cross-references and edit links pointing back to those source files.
oed_output = joinpath(@__DIR__, "src", "oed")
isdir(oed_output) && rm(oed_output; recursive = true)
mkpath(oed_output)
for page in ("index", "formulation", "designs", "numerics", "api", "references")
    source = joinpath(@__DIR__, "..", "lib", "CorleoneOED", "docs", "src", "$page.md")
    content = read(source, String)
    for guide in ("shooting", "initialization", "trajectories", "optimization_contracts", "automatic_differentiation")
        content = replace(
            content,
            "https://github.com/SciML/Corleone.jl/blob/main/docs/src/$guide.md" => "../$guide.md"
        )
    end
    edit_url = replace(relpath(source, oed_output), '\\' => '/')
    write(
        joinpath(oed_output, "$page.md"),
        "```@meta\nCurrentModule = CorleoneOED\nEditURL = \"$edit_url\"\n```\n\n" * content
    )
end
