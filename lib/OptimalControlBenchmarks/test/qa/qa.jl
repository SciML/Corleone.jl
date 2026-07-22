using SciMLTesting
using OptimalControlBenchmarks
using Test

run_qa(
    OptimalControlBenchmarks;
    ei_kwargs = (;
        # `problem_registry.jl` registers problems through a dynamic `include`,
        # which ExplicitImports cannot follow, so the module is unanalyzable for
        # the implicit/stale checks.
        no_implicit_imports = (; allow_unanalyzable = (OptimalControlBenchmarks,)),
        no_stale_explicit_imports = (; allow_unanalyzable = (OptimalControlBenchmarks,)),
        # `inputs` is re-exported by ModelingToolkit from ModelingToolkitBase.
        all_explicit_imports_via_owners = (; ignore = (:inputs,)),
    ),
    api_docs_kwargs = (;
        docs_src = normpath(@__DIR__, "..", "..", "..", "..", "docs", "src"),
    ),
)
