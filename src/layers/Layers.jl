using Reexport

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
using OhMyThreads: tmap
using Distributed: pmap

abstract type AbstractAutoShoot end


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

include("shooting_layer.jl")

export FixedLayer

export PiecewiseParameter
export inject!, reset!
export number_of_shooting_constraints
export shooting_constraints, shooting_constraints!
export get_timepoints

export Controls
export collect_timegrid

export ShootingInterval

export NoShoot, FixedShoot, AutoBlock

export ShootingLayer


#end
