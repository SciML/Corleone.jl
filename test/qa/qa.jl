using SciMLTesting
using Corleone
using Test

run_qa(
    Corleone;
    explicit_imports = true,
    # Corleone pulls all of its deps in with bare `using`, so the package leans
    # on a large set of implicit imports. Converting every one to an explicit
    # import is a sizable refactor tracked in SciML/Corleone.jl#103.
    ei_broken = (:no_implicit_imports,),
    ei_kwargs = (;
        # `ADTypes` is reached through `SciMLBase.ADTypes.AbstractADType`; ADTypes
        # is not a direct Corleone dependency, so the access goes via SciMLBase.
        all_qualified_accesses_via_owners = (; ignore = (:ADTypes,)),
        # Deliberate uses of well-known Base/stdlib/SciML internal names that are
        # not (yet) declared public in their owning modules.
        all_qualified_accesses_are_public = (;
            ignore = (
                :ADTypes, :AbstractDEAlgorithm, :AbstractDEProblem,
                :AbstractVecOrTuple, :EnsembleAlgorithm, :Fix1, :Fix2, :OneTo,
                :Splat, :Tunable, :canonicalize, :default_rng, :front,
                :get_colorizers, :initialparameters, :initialstates,
                :parameterlength, :setup, :tail,
            ),
        ),
    ),
)
