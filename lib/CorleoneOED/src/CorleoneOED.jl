module CorleoneOED

using Reexport
@reexport using Corleone
@reexport using Symbolics
using LuxCore
using Random
using SymbolicIndexingInterface
using SciMLStructures
using ForwardDiff
using SciMLBase
using DocStringExtensions
using LinearAlgebra

"""
    @symbolic_wrap(ex)

Reexported from Symbolics; wrap an expression as a symbolic value.
"""
var"@symbolic_wrap"

"""
    @wrapped(ex)

Reexported from Symbolics; create a wrapped symbolic expression.
"""
var"@wrapped"

"""
    RuleSet

Reexported from Symbolics; a collection of symbolic rewrite rules.
"""
RuleSet

"""
    get_canonical_expr(x)

Reexported from Symbolics; return the canonical symbolic expression for `x`.
"""
get_canonical_expr

"""
    infimum(x)

Reexported from Symbolics; return the lower endpoint or symbolic infimum of `x`.
"""
infimum

"""
    is_derivative(x)

Reexported from Symbolics; test whether `x` represents a symbolic derivative.
"""
is_derivative

"""
    istree(x)

Reexported from Symbolics; test whether `x` has tree-structured symbolic arguments.
"""
istree

"""
    solve_for(args...)

Reexported from Symbolics; solve a symbolic equation or system for selected variables.
"""
solve_for

"""
    supremum(x)

Reexported from Symbolics; return the upper endpoint or symbolic supremum of `x`.
"""
supremum

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
