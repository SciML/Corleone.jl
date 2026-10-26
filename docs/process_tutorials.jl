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


function make_tutorial(path)
    isfile(path) || return nothing
    metadata = extract_metadata(path)
    _..., scriptname = splitpath(path)
    fname = replace(lowercase(get(something(metadata, Dict()), "title", first(splitext(scriptname)))), " " => "_")
    outpath = joinpath(@__DIR__, "src", "examples")
    isdir(outpath) || mkdir(outpath)
    Literate.markdown(
        path, outpath,
        execute = true,
        preprocess = preprocess_script,
        flavor = Literate.CommonMarkFlavor(),
        name = fname
    )
    metadata = something(metadata, Dict("title" => fname))
    metadata["link"] = joinpath(".", "examples", fname)
    return metadata
end

# Only root-package examples using the current ShootingLayer API are built here.
# Sibling-library tutorials and the legacy ModelingToolkit facade are maintained
# separately from the root package manual.
example_output = joinpath(@__DIR__, "src", "examples")
isdir(example_output) && rm(example_output; recursive = true)

tutorials = map(
    [
        joinpath(@__DIR__, "..", "examples", "linear_quadratic", "main.jl"),
        joinpath(@__DIR__, "..", "examples", "lotka_fishing_optimal_control", "main.jl"),
    ]
) do tutorial
    make_tutorial(tutorial)
end
