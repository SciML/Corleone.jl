# Include the benchmark guide/API from its authoritative sublibrary source.
benchmark_output = joinpath(@__DIR__, "src", "benchmarks")
mkpath(benchmark_output)
benchmark_source = joinpath(
    @__DIR__, "..", "lib", "OptimalControlBenchmarks", "docs", "src", "index.md"
)
benchmark_edit_url = "https://github.com/SciML/Corleone.jl/blob/main/lib/OptimalControlBenchmarks/docs/src/index.md"
write(
    joinpath(benchmark_output, "index.md"),
    "```@meta\nCurrentModule = OptimalControlBenchmarks\nEditURL = \"$benchmark_edit_url\"\n```\n\n" *
        read(benchmark_source, String)
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
