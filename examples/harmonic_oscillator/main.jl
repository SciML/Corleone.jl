#src ---
#src title: The Harmonic Oscillator 
#src description: A super beginner-friendly guide to your optimal control problem
#src tags:
#src   - Beginner
#src   - Optimal Control
#src   - Linear System 
#src icon: 🌊
#src ---

# This is a quick intro based on [the lotka volterra fishing problem](https://mintoc.de/index.php?title=Lotka_Volterra_fishing_problem).

# ## Setup 
# We will use `Corleone` to define our optimal control problem. 
using Corleone

# Additionally, we will need the folllowing packages 
# - [`LuxCore`]() and [`Random`]() for basic setup functions
# - [`OrdinaryDiffEqTsit5`]() as an adaptive solver for the related ODEProblem
# - [`SymbolicIndexingInterface`]() to conviniently access variables and controls of the solution
# - [`Optimization`](), [`OptimizationMOI`](), [`Ipopt`](), and [`ComponentArrays`]() to setup and solve the optimization problem
# - [`CairoMakie`]() to plot the solution

using LuxCore
using Random
using OrdinaryDiffEqTsit5
using SymbolicIndexingInterface
using Optimization
using OptimizationMOI
using Ipopt
using ComponentArrays
using CairoMakie
