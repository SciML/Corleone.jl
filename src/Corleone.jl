module Corleone

using Reexport
using DocStringExtensions
using Random

using SciMLBase
using SciMLStructures
using Accessors
using LinearAlgebra

abstract type AbstractShootingProblem end 

include("single_shooting.jl")


end
