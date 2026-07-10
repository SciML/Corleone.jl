using Reexport 

#@reexport module Layers
using SymbolicIndexingInterface
using SciMLBase
using ConcreteStructs
using DocStringExtensions
using LuxCore
using SciMLStructures
using Random
using ChainRulesCore
using Functors

#include("../solutions/Solutions.jl")
#include("../core/Core.jl")



maybecallme(f::Base.Callable, ps, st) = f(ps, st)
maybecallme(x, ps, st) = x 
first_or_first(f::Base.Callable, ps, st) = first(f(ps, st))
first_or_first(f::Tuple, ps, st) = maybecallme(first(f), ps, st)
last_or_last(f::Base.Callable, ps, st) = last(f(ps, st))
last_or_last(f::Tuple, ps, st) = maybecallme(Base.last(f), ps, st)


include("abstract.jl")

include("piecewise_constant.jl")

include("parameter_container.jl")

export PiecewiseParameter
export number_of_shooting_constraints
export shooting_constraints, shooting_constraints!

export Controls


#end