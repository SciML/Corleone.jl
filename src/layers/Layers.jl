using Reexport 

#@reexport module Layers
using SparseArrays
using SymbolicIndexingInterface
using SciMLBase
using ConcreteStructs
using DocStringExtensions
using LuxCore
using SciMLStructures
using Random
using ChainRulesCore
using Functors

abstract type AbstractAutoShoot end

#include("../solutions/Solutions.jl")
#include("../core/Core.jl")



maybecallme(f::Base.Callable, args...) = f(args...)
maybecallme(args...) = first(args) 
first_or_first(f::Base.Callable, ps, st) = first(f(ps, st))
first_or_first(f::Tuple, ps, st) = maybecallme(first(f), ps, st)
last_or_last(f::Base.Callable, ps, st) = last(f(ps, st))
last_or_last(f::Tuple, ps, st) = maybecallme(Base.last(f), ps, st)


include("abstract.jl")

include("piecewise_constant.jl")

include("parameter_container.jl")

include("shooting.jl")

include("shooting_interval.jl")

include("parallel_layer.jl")

include("shooting_layer.jl")

export PiecewiseParameter
export number_of_shooting_constraints
export shooting_constraints, shooting_constraints!

export Controls

export ShootingInterval

export ShootingLayer


#end