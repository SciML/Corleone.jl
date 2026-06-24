module CorleoneOED

using Reexport
@reexport using Corleone
# Symbolics (through at least v7.28.1) still lists `Variable` in its `export`s even though that
# binding was removed, so a blanket `@reexport using Symbolics` re-exports an undefined name
# into CorleoneOED (flagged by Aqua's undefined-exports check). Reproduce Reexport's behaviour
# (`using` + re-export the package's exported names) while dropping the names that no longer
# resolve in Symbolics.
# Fixed upstream in JuliaSymbolics/Symbolics.jl#1906; once that is released, drop this loop,
# bump the Symbolics compat floor accordingly, and restore `@reexport using Symbolics`.
using Symbolics
for name in Reexport.exported_names(Symbolics)
    isdefined(Symbolics, name) || continue
    @eval export $name
end
using LuxCore
using Random
using SymbolicIndexingInterface
using SciMLStructures
using ForwardDiff
using SciMLBase
using DocStringExtensions
using LinearAlgebra

# TODO Clean and more tests (also for LV)
include("augmentation.jl")

# TODO Docs
include("oed.jl")
export OEDLayer
export fisher_information, observed_equations, sensitivities
export local_information_gain, global_information_gain

include("multiexperiments.jl")
export MultiExperimentLayer

# TODO
# Dispatch for optimality(crit, oed, x, ps, st)
# --> Compute optimality condition for specific crit
# e.g. tr(\Pi) for the ACriterion
include("criteria.jl")
export ACriterion, DCriterion, ECriterion
export FisherACriterion, FisherECriterion, FisherDCriterion


end
