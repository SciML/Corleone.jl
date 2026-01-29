using Pkg

Pkg.develop(path = joinpath(@__DIR__, ".."))

using Documenter, Corleone
using DocumenterVitepress
using Literate
using YAML, JSON

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


function make_tutorial(path)
    isfile(path) || return nothing
    activate_env = dirname(path)
    Pkg.activate(activate_env)
    metadata = extract_metadata(path)
    _..., scriptname = splitpath(path)
    fname = replace(lowercase(get(metadata, "title", first(splitext(scriptname)))), " " => "_")
    outpath = joinpath(@__DIR__, "src", "examples")
    isdir(outpath) || mkdir(outpath)
    Literate.markdown(path, outpath, execute = false, flavor = Literate.CommonMarkFlavor(), name = fname)
    Pkg.activate(@__DIR__)
    @info Pkg.status()
    metadata["link"] = joinpath(".", "examples", fname)
    @info metadata
    return metadata
end

tutorials = map(
    [
        abspath("examples/harmonic_oscillator/main.jl"),
        abspath("examples/lotka_fishing_optimal_control/main.jl"),
    ]
) do tutorial
    make_tutorial(tutorial)
end

function generate_searchable_index(tutorials)
    data = filter(!isnothing, tutorials)
    output_path = joinpath(@__DIR__, "src", "tutorials.md")
    json_path = joinpath(@__DIR__, "src", "public", "tutorial_data.json")
    write(json_path, JSON.json(data))

    output = """
    ```@raw html
    <script setup lang="ts">
    import Gallery from "./components/Gallery.vue";
    import data from "./public/tutorial_data.json"; 
    </script>

    # Tutorials

    <Gallery :items="data" />
    ``` 
    """
    
    write(output_path, output)
end

generate_searchable_index(tutorials)

makedocs(
    sitename = "Corleone.jl",
    authors = "Carl Julius Martensen, Christoph Plate, et al.",
    modules = [Corleone],
    format = DocumenterVitepress.MarkdownVitepress(
        repo = "github.com/SciML/Corleone.jl",
        devbranch = "main", # or master, trunk, ...
        devurl = "dev",
        # if you use something else than yourname.github.io/YourPackage.jl
    ),
    pages = [
        "Home" => "index.md",
        "Tutorials" => "tutorials.md", #tutorials,
        #"Examples" => [#"Optimal Control" => "./examples/lotka.md",
        #"Multiple Shooting" => "./examples/multiple_shooting.md",
        # "Optimal Experimental Design" => "./examples/lotka_oed.md",
        #"Multiexperiments" => "./examples/multiexperiments.md"
        #],
        # "API Reference" => "api.md",
    ],
    checkdocs = :none,
    remotes = nothing
)
