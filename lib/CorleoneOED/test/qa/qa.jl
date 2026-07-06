using SciMLTesting
using CorleoneOED
using Test

run_qa(
    CorleoneOED;
    explicit_imports = true,
    # CorleoneOED pulls Corleone and its other deps in with bare `using`, so it
    # leans on a large set of implicit imports. Converting every one to an
    # explicit import is a sizable refactor tracked in SciML/Corleone.jl#103.
    ei_broken = (:no_implicit_imports,),
    ei_kwargs = (;
        # Names still not declared public in their owning modules: SciMLBase
        # internals (`AbstractDEAlgorithm`, `AbstractDEProblem`, `get_colorizers`),
        # SciMLStructures internals (`Tunable`, `canonicalize`), SymbolicUtils
        # internal (`Code`), Symbolics internals (`getdefaultval`, `setdefaultval`,
        # `variables`), ForwardDiff internal (`jacobian`), and Corleone's own
        # as-yet-unexported helpers reached through `Corleone.*`.
        all_qualified_accesses_are_public = (;
            ignore = (
                :AbstractDEAlgorithm, :AbstractDEProblem, :Code, :Tunable,
                :build_index_grid, :canonicalize, :get_block_structure,
                :get_bounds, :get_colorizers,
                :get_number_of_shooting_constraints, :get_timegrid, :getdefaultval,
                :jacobian, :retrieve_symbol_cache, :setdefaultval,
                :shooting_constraints, :shooting_constraints!, :variables,
            ),
        ),
    ),
)
