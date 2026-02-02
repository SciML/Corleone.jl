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
    content = replace(content, "TODAY" => Date(now()))
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
    activate_env = dirname(path)
    Pkg.activate(activate_env)
    Pkg.develop(path = joinpath(@__DIR__, ".."))
    Pkg.instantiate()
    metadata = extract_metadata(path)
    _..., scriptname = splitpath(path)
    fname = replace(lowercase(get(metadata, "title", first(splitext(scriptname)))), " " => "_")
    outpath = joinpath(@__DIR__, "src", "examples")
    isdir(outpath) || mkdir(outpath)
    Literate.markdown(
        path, outpath,
        execute = true,
        preprocess = preprocess_script,
        # flavor = Literate.CommonMarkFlavor(),
        name = fname
    )
    Pkg.activate(@__DIR__)
    metadata["link"] = joinpath(".", "examples", fname)
    return metadata
end

tutorials = map(
    [
        abspath("examples/linear_quadratic/main.jl"),
        #   abspath("examples/lotka_fishing_optimal_control/main.jl"),
    ]
) do tutorial
    make_tutorial(tutorial)
end

function generate_searchable_index(tutorials)
    data = filter(!isnothing, tutorials)
    output_path = joinpath(@__DIR__, "src", "tutorials.md")
    json_path = joinpath(@__DIR__, "src", "assets", "tutorial_data.json")
    write(json_path, JSON.json(data))

    output = """
    ```@raw html
    <script setup lang="ts">
    import Gallery from "./components/Gallery.vue";
    import data from "./assets/tutorial_data.json"; 
    </script>

    # Tutorials

    <Gallery :items="data" />
    ``` 
    """

    return write(output_path, output)
end

generate_searchable_index(tutorials)
