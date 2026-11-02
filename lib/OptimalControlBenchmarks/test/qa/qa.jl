using SciMLTesting
using OptimalControlBenchmarks
using Test

run_qa(
    OptimalControlBenchmarks;
    api_docs_kwargs = (; docs_src = joinpath(@__DIR__, "..", "..", "docs", "src")),
    ei_kwargs = (;
        # The registry discovers and includes problem files at package load time;
        # ExplicitImports cannot statically follow these generated include paths.
        no_implicit_imports = (; allow_unanalyzable = (OptimalControlBenchmarks,)),
        no_stale_explicit_imports = (; allow_unanalyzable = (OptimalControlBenchmarks,)),
    ),
)
