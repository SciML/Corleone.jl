using Test, Dates, TOML, SHA, Base64, DemoCards, YAML
include("../tutorial_cache.jl")
using .TutorialCache

const REPOSITORY = dirname(dirname(@__DIR__))
const PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII="

function write_tutorial(root, name; dependencies = String[])
    directory = joinpath(root, "examples", name)
    mkpath(directory)
    dependency_yaml = isempty(dependencies) ? "" :
        "# cache_dependencies:\n" * join(["#   - $path\n" for path in dependencies])
    source = """
    # ---
    # title: Tutorial $name
    # description: Cache integration fixture.
    # category: optimal_control
    # id: fixture_$name
    # notebook: false
    # generate_cover: false
    $dependency_yaml# ---

    using Base64
    open($(repr(joinpath(root, "executions.txt"))), "a") do io
        println(io, $(repr(name)))
    end
    struct FixturePlot end
    Base.show(io::IO, ::MIME"text/html", ::FixturePlot) = print(io, $(repr(string("<img src=", Char(34), "data:image/png;base64,", PNG, Char(34), ">"))))
    FixturePlot()
    """
    return write(joinpath(directory, "main.jl"), source)
end

function fixture(root)
    docs = joinpath(root, "docs")
    mkpath(joinpath(docs, "src"))
    mkpath(joinpath(root, "src"))
    for file in ("process_tutorials.jl", "tutorial_cache.jl")
        cp(joinpath(REPOSITORY, "docs", file), joinpath(docs, file))
    end
    write(joinpath(root, "Project.toml"), "name = \"Fixture\"\n")
    write(joinpath(root, "src", "model.jl"), "model = 1\n")
    write(joinpath(docs, "Project.toml"), "[deps]\n")
    write(joinpath(docs, "Manifest.toml"), "manifest_format = \"2.0\"\n")
    write(
        joinpath(docs, "tutorials.toml"), """
        [[tutorials]]
        source = "examples/first/main.jl"
        name = "first"
        [[tutorials]]
        source = "examples/second/main.jl"
        name = "second"
        """
    )
    mkpath(joinpath(root, "examples", "shared"))
    write(joinpath(root, "examples", "shared", "helper.jl"), "helper = 1\n")
    write_tutorial(root, "first")
    write_tutorial(root, "second"; dependencies = ["../shared/helper.jl"])
    # The real pipeline also copies the sublibrary manuals into the site.
    for (library, pages) in (
            "CorleoneOED" => ("index", "formulation", "designs", "numerics", "api", "references"),
            "CorleoneInverse" => ("index", "observations", "experiments", "api"),
        )
        directory = joinpath(root, "lib", library, "docs", "src")
        mkpath(directory)
        for page in pages
            write(joinpath(directory, "$page.md"), "# $page\n")
        end
    end
    return docs
end

function build_fixture(docs; draft = false, cache = true)
    sandbox = Module(gensym(:DocsBuild))
    Core.eval(sandbox, :(using Dates, DemoCards, YAML))
    Core.eval(sandbox, :(include(file) = Base.include($sandbox, file)))
    Core.eval(sandbox, :(draft = $draft))
    return withenv("DOCS_TUTORIAL_CACHE" => string(cache)) do
        Base.include(sandbox, joinpath(docs, "process_tutorials.jl"))
        Core.eval(sandbox, :tutorials)
    end
end

executions(root) = readlines(joinpath(root, "executions.txt"))

@testset "Independent tutorial execution cache" begin
    mktempdir() do root
        docs = fixture(root)
        output = joinpath(docs, "src", "examples")
        first_build = build_fixture(docs)
        @test executions(root) == ["first", "second"]
        @test all(!metadata["cache_hit"] for metadata in first_build)
        original = read(joinpath(output, "first.md"), String)
        @test occursin("first-figure-1.png", original)
        @test isfile(joinpath(output, "first.jl"))
        @test isfile(joinpath(docs, "src", "gallery", "covers", "first.png"))

        warm_build = build_fixture(docs)
        @test all(metadata["cache_hit"] for metadata in warm_build)
        @test executions(root) == ["first", "second"]
        @test read(joinpath(output, "first.md"), String) == original
        @test read(joinpath(output, "first-figure-1.png")) == base64decode(PNG)
        gallery = read(joinpath(docs, "src", "gallery", "index.md"), String)
        @test occursin("covers/first.png", gallery) && occursin("covers/second.png", gallery)

        # Editing one example or its data must preserve the other example's hit.
        open(io -> write(io, "\n# Edited tutorial\n"), joinpath(root, "examples", "first", "main.jl"), "a")
        build_fixture(docs)
        @test executions(root) == ["first", "second", "first"]
        write(joinpath(root, "examples", "first", "measurements.tsv"), "time\ty\n0\t1\n")
        build_fixture(docs)
        @test executions(root)[end] == "first"
        @test length(executions(root)) == 4
        write(joinpath(root, "examples", "shared", "helper.jl"), "helper = 2\n")
        build_fixture(docs)
        @test executions(root)[end] == "second"
        @test length(executions(root)) == 5

        # Handwritten pages do not change executed results; model changes do.
        write(joinpath(docs, "src", "manual.md"), "New prose\n")
        build_fixture(docs)
        @test length(executions(root)) == 5
        write(joinpath(root, "src", "model.jl"), "model = 2\n")
        build_fixture(docs)
        @test executions(root)[(end - 1):end] == ["first", "second"]
        @test length(executions(root)) == 7
        write(joinpath(docs, "Manifest.toml"), "manifest_format = \"2.0\"\nproject_hash = \"changed\"\n")
        build_fixture(docs)
        @test length(executions(root)) == 9

        # Missing/corrupt assets rerun only their owning tutorial.
        rm(joinpath(docs, ".cache", "tutorials", "first", "first-figure-1.png"))
        build_fixture(docs)
        @test length(executions(root)) == 10
        @test executions(root)[end] == "first"
        write(joinpath(docs, ".cache", "tutorials", "second", "second.md"), "corrupt")
        build_fixture(docs)
        @test length(executions(root)) == 11
        @test executions(root)[end] == "second"

        build_fixture(docs; draft = true)
        @test length(executions(root)) == 11
        build_fixture(docs)
        @test length(executions(root)) == 11 # Draft never poisons executed entries.
        build_fixture(docs; cache = false)
        @test length(executions(root)) == 13

        # Registry edits reuse existing entries and remove obsolete pages/covers.
        registry = joinpath(docs, "tutorials.toml")
        entries = TOML.parsefile(registry)
        pop!(entries["tutorials"])
        open(io -> TOML.print(io, entries), registry, "w")
        build_fixture(docs)
        @test length(executions(root)) == 13
        @test !isfile(joinpath(output, "second.md"))
        @test !isfile(joinpath(docs, "src", "gallery", "covers", "second.png"))
        write_tutorial(root, "third")
        push!(entries["tutorials"], Dict("source" => "examples/third/main.jl", "name" => "third"))
        open(io -> TOML.print(io, entries), registry, "w")
        build_fixture(docs)
        @test length(executions(root)) == 14
        @test executions(root)[end] == "third"

        # Failed execution must not publish a reusable cache entry.
        write_tutorial(root, "failure")
        open(
            io -> write(io, "\nerror(\"intentional fixture failure\")\n"),
            joinpath(root, "examples", "failure", "main.jl"), "a"
        )
        push!(entries["tutorials"], Dict("source" => "examples/failure/main.jl", "name" => "failure"))
        open(io -> TOML.print(io, entries), registry, "w")
        @test_throws LoadError build_fixture(docs)
        @test !isdir(joinpath(docs, ".cache", "tutorials", "failure"))
        write_tutorial(root, "failure")
        build_fixture(docs)
        @test isfile(joinpath(docs, ".cache", "tutorials", "failure", "entry.toml"))

        @test_throws ArgumentError tutorial_key(
            joinpath(root, "examples", "first", "main.jl"), "key";
            dependencies = ["missing.jl"]
        )
        @test_throws ArgumentError valid_tutorial_cache(joinpath(docs, ".cache"), "../escape", "key")
        @test !valid_tutorial_cache(joinpath(docs, ".cache"), "missing", "key")
    end
end

@testset "Cache fingerprints and artifact ownership" begin
    mktempdir() do root
        docs = fixture(root)
        manifest = joinpath(docs, "Manifest.toml")
        write(manifest, "manifest_format = \"2.0\"\n[[deps.Corleone]]\npath = \"/old/checkout\"\n")
        context = tutorial_context(root)
        write(manifest, "manifest_format = \"2.0\"\n[[deps.Corleone]]\npath = \"/new/checkout\"\n")
        @test tutorial_context(root) == context
        key = tutorial_key(joinpath(root, "examples", "first", "main.jl"), context; root)
        cp(joinpath(root, "examples", "first"), joinpath(root, "examples", "renamed"))
        @test tutorial_key(joinpath(root, "examples", "renamed", "main.jl"), context; root) != key
        output = mktempdir(root)
        for file in ("first.md", "first.jl", "first-figure-1.png", "first_other-figure-1.png")
            write(joinpath(output, file), "fixture")
        end
        cache = joinpath(root, "cache")
        save_tutorial_cache!(cache, "first", key, output)
        @test valid_tutorial_cache(cache, "first", key)
        @test !isfile(joinpath(cache, "first", "first_other-figure-1.png"))
        @test !valid_tutorial_cache(cache, "first", "different inputs")
    end
end
