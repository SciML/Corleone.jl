using SciMLTesting
using Corleone
using Test

# Load the numerical extensions so ExplicitImports checks their methods too.
# Plotting has a separate environment; ModelingToolkit integration is deferred.
using ComponentArrays, Zygote

@testset "Extensions loaded" begin
    for ext in (:CorleoneComponentArraysExtension, :CorleoneZygoteExtension)
        @test Base.get_extension(Corleone, ext) !== nothing
    end
end

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
        # The Zygote extension implements these private parent-package hooks.
        all_explicit_imports_are_public = (;
            ignore = (:is_extension_loaded, :unwrapbuffer, :wrapbuffer),
        ),
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
                # SII timeseries dispatch hook, not declared public upstream.
                :parameter_values_at_time,
                # Zygote's documented mutation buffer is not declared public.
                :Buffer,
                # Corleone's own internals. The extensions exist to add methods to
                # these, which is only expressible as a qualified access into the
                # parent package; they are internal hooks, not user API.
                :AbstractCorleoneFunctionWrapper, :get_number_of_shooting_constraints,
                :retrieve_symbol_cache, :shooting_constraints!, :to_vec,
                :wrap_functions, :replace_parameter_fields, :parameter_bound_tree, :unbounded_parameter_bounds, :empty_parameter_structure,
                # Symbolics/SymbolicUtils term-rewriting internals used to normalize
                # MTK constraints and read symbolic defaults; no public spelling.
                :Code, :canonical_form, :getdefaultval,
                # Makie's recipe hooks, which every plot recipe must implement even
                # though they are not declared `public`.
                :plotsym, :plottype,
            ),
        ),
    ),
)
