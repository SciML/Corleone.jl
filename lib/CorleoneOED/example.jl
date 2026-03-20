using Corleone
using CorleoneOED
using SymbolicIndexingInterface
using OrdinaryDiffEqTsit5

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
continuous_observed = ContinuousMeasurement(ControlParameter(0.:1.:2., name = :w2), (u, p, t) -> p[1] + u[1]^2)

# The function here adds them to the system 
CorleoneOED.add_observed!(symbolic_system, discrete_observed, continuous_observed)

# Last a new DEProblem and layer is defined with all old controls and new controls and measurements 
new_layer = SingleShootingLayer(symbolic_system, layer)

# The new layer has all the necessary differential equations 
# This is wrapped into the OEDLayer which return the trajectory but computes also the discrete measurements and returns the Fisher as the sum 
oed_layer = OEDLayer(symbolic_system, new_layer)
# The OEDLayer contains the new_layer, the discrete_observed controls and so on. 

