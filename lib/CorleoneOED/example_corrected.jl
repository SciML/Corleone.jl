using Corleone
using CorleoneOED
using SymbolicIndexingInterface
using OrdinaryDiffEqTsit5

using LuxCore
using Random

# Define a Problem using Corleone 
function f(du, u, p, t)
    du[1] = -p[1] * u[1]
end
u0 = [1.0]
tspan = (0.0, 2.0)
p = [0.5]

prob = ODEProblem{true}(
    ODEFunction(f, sys=SymbolCache([:x], [:p], :t)), u0, tspan, p)
control = Corleone.ControlParameter(0.0:0.5:2.0, name=:p)
layer = Corleone.SingleShootingLayer(prob, control; algorithm=Tsit5())

# Now the inner functions 

# Should return the symbolic form of the equations (so f(du, u, p, t) but symbolically evaluated) 
# Should use the system. If the system is a SymbolCache, generate Nums based on the Symbols. Make this a dispatch function 
# similar to Corleone.default_system(layer, controls) 
symbolic_system = CorleoneOED.get_symbolic_equations(layer) 

# This function appends the forward sensitivity to the eom based on the parameter values of the system
# This means a new set of differential equations and states.
CorleoneOED.append_sensitivity!(symbolic_system)

# The next step parses in discrete measurements (Fisher_contribution is gramian of  w1 * dmeasure/ dx * G)
discrete_observed = DiscreteMeasurement(ControlParameter(0.:1.:2., name = :w1), (u, p, t) -> u[1]^2)

# This is a continuous measurement, which is dFisher / dt = gramian of :w2 * dmeasure / dx * G which is added to the differential equations
continuous_observed = ContinuousMeasurement(ControlParameter(0.:1.:2., name = :w2), (u, p, t) -> p[1] * u[1])

# The function here adds them to the system 
CorleoneOED.add_observed!(symbolic_system, discrete_observed, continuous_observed)

# Here the symbolic system should have the following entries for the continuous fisher 
# dF_cont/ dt = w2 * G_1,1 ^ 2 * x ^ 2 


# And for the discrete 
# F_discrete = w1 * G_1,1 ^ 2 * p[1]^2  

# Last a new DEProblem and layer is defined with all old controls and new controls and measurements 
new_layer = SingleShootingLayer(symbolic_system, layer)
# The new_layer must contain 2 controls :p and :w2
# Check if ps contains also :w2 as a new control for the fisher information, given that it is 
# used for a continuous measurement
ps, st = LuxCore.setup(Random.default_rng(), new_layer) 
traj, _ = new_layer(nothing, ps, st)

# The new layer has all the necessary differential equations 
# This is wrapped into the OEDLayerV2 which return the trajectory but computes also the discrete measurements and returns the Fisher as the sum 
# This layer should only be named OEDLayer and contain the fields 
# layer:: The singel shooting layer ( with the augmentations ) 
# controls:: The control parameter which represent the discrete observations 
# discrete_fisher:: The definition of the discrete fisher useable as a getsym(problem, F_discrete) 
# Hence it is a LuxCore.AbstractLuxContainerLayer
oed_layer = OEDLayerV2(symbolic_system, new_layer)
# Check if ps contains also :w2 as a new control for the continuous fisher information
# And :w1 as a discrete control for the discrete fisher information
ps, st = LuxCore.setup(Random.default_rng(), oed_layer) 
# This layer returns the fisher = discrete_fisher + continuous_fisher and the trajectory
(fisher, trajectory), _ = oed_layer(nothing, ps, st)

println("Example completed successfully!")
println("Fisher information: ", fisher)
