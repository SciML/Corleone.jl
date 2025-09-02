module CorleoneCore

using Reexport
using DocStringExtensions
using Random

using ComponentArrays

using SciMLBase
using CommonSolve
using ChainRulesCore

using LuxCore

include("local_controls.jl")
export ControlParameter

# Defines approximators for signals
# TODO Add Linear, Quadratic and stuff here

#include("approximators/piecewiseconstant.jl")
#export PiecewiseConstant
#include("approximators/container.jl")
#export SignalContainer 

#include("wrapper.jl")
#export StatefulWrapper

#include("dynamic/controlled.jl")
#export ControlledDynamics

#include("dynamic/shooting.jl")
#export  ShootingProblem

#include("dynamics.jl")
#export DynamicsFunction

end
