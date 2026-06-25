using SciMLTesting
using OptimalControlBenchmarks
using Test

run_qa(
    OptimalControlBenchmarks;
    explicit_imports = true,
    ei_kwargs = (;
        # `problem_registry.jl` registers problems through a dynamic `include`,
        # which ExplicitImports cannot follow, so the module is unanalyzable for
        # the implicit/stale checks.
        no_implicit_imports = (; allow_unanalyzable = (OptimalControlBenchmarks,)),
        no_stale_explicit_imports = (; allow_unanalyzable = (OptimalControlBenchmarks,)),
        # `inputs` is re-exported by ModelingToolkit from ModelingToolkitBase.
        all_explicit_imports_via_owners = (; ignore = (:inputs,)),
        all_explicit_imports_are_public = (; ignore = (:inputs,)),
        # Deliberate uses of internal names not (yet) declared public upstream.
        all_qualified_accesses_are_public = (; ignore = (:default_rng, :initialstates)),
    ),
)
