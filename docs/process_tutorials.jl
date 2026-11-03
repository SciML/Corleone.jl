using Base64: base64decode

function extract_metadata(file_path)
    content = read(file_path, String)
    # Read DemoCards' native commented YAML frontmatter.
    m = match(r"\A# ---(?:\r?\n)((?:# .*\r?\n)*)# ---", content)

    if m !== nothing
        # Remove the comment prefix to obtain YAML.
        yaml_text = replace(m.captures[1], r"^# "m => "")
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
    content = replace(content, "TODAY" => string(Date(now())))
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


# DemoCards generates the page heading and anchor from native frontmatter.
# Keep that heading, and point the source badge at the authoritative script.
function normalize_demo_output(content)
    edit_url = match(r"EditURL = \"([^\"]+)\"", content).captures[1]
    source_url = replace(replace(edit_url, "github.com" => "raw.githubusercontent.com"), "/blob/" => "/")
    badge_link = r"(\[!\[Source code\]\([^\n)]*\)\]\()[^)]+(\))"
    return replace(
        content, badge_link => m -> begin
            captures = match(badge_link, m).captures
            captures[1] * source_url * captures[2]
        end
    )
end

function make_tutorial(path; name = nothing, draft = false)
    isfile(path) || return nothing
    metadata = extract_metadata(path)
    _..., scriptname = splitpath(path)
    title = get(something(metadata, Dict()), "title", first(splitext(scriptname)))
    fname = isnothing(name) ? replace(lowercase(title), " " => "_") : name
    outpath = joinpath(@__DIR__, "src", "examples")
    isdir(outpath) || mkdir(outpath)
    category = get(
        something(metadata, Dict()), "category",
        occursin("CorleoneOED", path) ? "experimental_design" : "optimal_control"
    )
    staged = joinpath(demo_source, category, "$fname.jl")
    content = preprocess_script(read(path, String))
    # Each tutorial disables cover/notebook execution in its native frontmatter.
    # Only the documentation build's draft mode needs a staging-time override.
    content = replace(content, "# ---\n" => "# ---\n# execute: $(!draft)\n"; count = 1)
    write(staged, content)
    metadata = something(metadata, Dict("title" => fname))
    metadata["source"] = path
    metadata["link"] = joinpath(".", "examples", fname)
    return metadata
end

# Both sites execute the same authoritative OED examples. Explicit names preserve
# the OED tutorial URLs already published by main, independently of source titles.
example_output = joinpath(@__DIR__, "src", "examples")
isdir(example_output) && rm(example_output; recursive = true)

demo_source = joinpath(@__DIR__, ".democards", "gallery")
isdir(demo_source) && rm(demo_source; recursive = true)
for (folder, title) in (("optimal_control", "Optimal Control"), ("experimental_design", "Optimal Experimental Design"), ("parameter_estimation", "Parameter Estimation"))
    mkpath(joinpath(demo_source, folder))
    write(joinpath(demo_source, folder, "config.json"), "{\"title\":\"$title\"}")
end
write(joinpath(demo_source, "config.json"), "{\"title\":\"Tutorials\",\"theme\":\"bokehlist\",\"order\":[\"optimal_control\",\"experimental_design\",\"parameter_estimation\"]}")

tutorials = map(
    [
        (joinpath(@__DIR__, "..", "examples", "linear_quadratic", "main.jl"), nothing),
        (joinpath(@__DIR__, "..", "examples", "lotka_fishing_optimal_control", "main.jl"), nothing),
        (joinpath(@__DIR__, "..", "examples", "mtk_lotka", "main.jl"), nothing),
        (joinpath(@__DIR__, "..", "examples", "lotka_universal_differential_equation", "main.jl"), nothing),
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
    make_tutorial(path; name, draft)
end

# DemoCards owns Julia/Literate generation; retain published tutorial URLs.
demo_pages, demo_postprocess, demo_assets = DemoCards.makedemos(
    demo_source; root = @__DIR__, credit = false, throw_error = true,
    edit_branch = "main",
)
generated = joinpath(@__DIR__, "src", "gallery")
for folder in ("optimal_control", "experimental_design", "parameter_estimation")
    for file in readdir(joinpath(generated, folder); join = true)
        mv(file, joinpath(example_output, basename(file)); force = true)
    end
end
# Keep DemoCards' metadata-driven list while preserving published tutorial URLs.
for metadata in tutorials
    name = basename(metadata["link"])
    path = joinpath(example_output, "$name.md")
    edit_url = "https://github.com/SciML/Corleone.jl/blob/main/" * replace(relpath(metadata["source"], joinpath(@__DIR__, "..")), '\\' => '/')
    content = tutorial_figures(read(path, String), metadata["title"], example_output, name)
    content = replace(content, r"```@meta\n.*?```\n"s => ""; count = 1)
    write(path, normalize_demo_output("```@meta\nEditURL = \"$edit_url\"\n```\n\n" * content))
end

# Reuse executed trajectory figures rather than running examples for covers.
function trajectory_covers(tutorials, example_output, generated; draft = false)
    gallery_path = joinpath(generated, "index.md")
    gallery = read(gallery_path, String)
    for metadata in tutorials
        name = basename(metadata["link"])
        figure = get(metadata, "cover_figure", 1)
        image = joinpath(example_output, "$name-figure-$figure.png")
        if !isfile(image)
            draft && continue
            error("Missing trajectory cover for $name: $image")
        end
        id = metadata["id"]
        cp(image, joinpath(generated, "covers", "$name.png"); force = true)
        gallery = replace(
            gallery,
            "![bokeh-list-card-cover-image](covers/democards_logo.svg)](@ref $id)" =>
                "![Trajectory plot for $(metadata["title"])](covers/$name.png)](@ref $id)"
        )
    end
    return write(gallery_path, gallery)
end
trajectory_covers(tutorials, example_output, generated; draft)

# Include the benchmark guide/API from its authoritative sublibrary source.
benchmark_output = joinpath(@__DIR__, "src", "benchmarks")
mkpath(benchmark_output)
benchmark_source = joinpath(
    @__DIR__, "..", "lib", "OptimalControlBenchmarks", "docs", "src", "index.md"
)
benchmark_edit_url = "https://github.com/SciML/Corleone.jl/blob/main/lib/OptimalControlBenchmarks/docs/src/index.md"
benchmark_content = replace(
    read(benchmark_source, String),
    (
        "https://github.com/SciML/Corleone.jl/blob/main/docs/src/$guide.md" => "../$guide.md"
            for guide in ("shooting", "optimization_contracts")
    )...
)
write(
    joinpath(benchmark_output, "index.md"),
    "```@meta\nCurrentModule = OptimalControlBenchmarks\nEditURL = \"$benchmark_edit_url\"\n```\n\n" *
        benchmark_content
)

benchmark_catalog = joinpath(@__DIR__, "..", "lib", "OptimalControlBenchmarks", "docs", "src", "problems.md")
write(
    joinpath(benchmark_output, "problems.md"),
    "```@meta\nEditURL = \"https://github.com/SciML/Corleone.jl/blob/main/lib/OptimalControlBenchmarks/docs/src/problems.md\"\n```\n\n" *
        read(benchmark_catalog, String)
)

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
