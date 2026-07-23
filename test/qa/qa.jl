using SciMLTesting
using Corleone
using Test

run_qa(
    Corleone;
    # Corleone pulls all of its deps in with bare `using`, so the package leans
    # on a large set of implicit imports. Converting every one to an explicit
    # import is a sizable refactor tracked in SciML/Corleone.jl#103.
    ei_broken = (:no_implicit_imports,),
    ei_kwargs = (;
        # `ADTypes` is reached through `SciMLBase.ADTypes.AbstractADType`; ADTypes
        # is not a direct Corleone dependency, so the access goes via SciMLBase.
        all_qualified_accesses_via_owners = (; ignore = (:ADTypes,)),
        # Names still not declared public in their owning modules: Base internals
        # (`AbstractVecOrTuple`, `Splat`), SciMLBase internals (`ADTypes`,
        # `AbstractDEAlgorithm`, `AbstractDEProblem`, `EnsembleAlgorithm`,
        # `get_colorizers`), and SciMLStructures internals (`Tunable`,
        # `canonicalize`).
        all_qualified_accesses_are_public = (;
            ignore = (
                :ADTypes, :AbstractDEAlgorithm, :AbstractDEProblem,
                :AbstractVecOrTuple, :EnsembleAlgorithm, :Splat, :Tunable,
                :canonicalize, :get_colorizers,
            ),
        ),
    ),
)
