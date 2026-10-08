using CorleoneBase
using SciMLTesting
using Test

@test realpath(pkgdir(CorleoneBase)) == realpath(joinpath(@__DIR__, "../.."))
@test Base.get_extension(CorleoneBase, :CorleoneBaseZygoteExtension) === nothing
using ChainRulesCore
@test Base.get_extension(CorleoneBase, :CorleoneBaseZygoteExtension) === nothing
using Zygote
@test Base.get_extension(CorleoneBase, :CorleoneBaseZygoteExtension) !== nothing

run_qa(CorleoneBase;
    # This sublibrary has no standalone manual; retain docstring checks.
    api_docs_kwargs = (; rendered = false),
    ei_kwargs = (;
        # Zygote documents @adjoint as its API, but its defining owner is
        # ZygoteRules. Do not add another weakdep solely for this reexport.
        all_qualified_accesses_via_owners = (; ignore = (Symbol("@adjoint"),)),
        # Base has no public type-preserving NamedTuple key subtraction API.
        # Zygote documents @adjoint and Buffer but does not mark them public;
        # Core.kwcall is the existing keyword-init AD rule's dispatch hook.
        all_qualified_accesses_are_public = (;
            ignore = (:structdiff, Symbol("@adjoint"), :Buffer, :kwcall),
        ),
        # These private parent-package hooks implement the optional AD rules;
        # making them public merely for the extension would expand the API.
        all_explicit_imports_are_public = (;
            ignore = (:AbstractSequentialProblem, :SequentialProblemIterator,
                :get_problem, :make_buffer, :prepare_stage_problem, :terminal, :transition),
        ),
    ),
)
