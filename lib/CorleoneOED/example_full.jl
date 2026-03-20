using Corleone
using CorleoneOED
using SymbolicIndexingInterface
using OrdinaryDiffEqTsit5
using SciMLBase
using Random
using LuxCore

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
println("Step 1: Extracting symbolic equations...")
symbolic_system = CorleoneOED.get_symbolic_equations(layer) 
println("  System has $(length(symbolic_system.vars)) states and $(length(symbolic_system.parameters)) parameters")

println("\nStep 2: Adding sensitivities...")
CorleoneOED.append_sensitivity!(symbolic_system)
println("  Sensitivities added for parameters: $(symbolic_system.sensitivity_params)")
println("  Sensitivity matrix size: $(size(symbolic_system.sensitivities))")

println("\nStep 3: Adding measurements...")
discrete_observed = DiscreteMeasurement(ControlParameter(0.:1.:2., name = :w1), (u, p, t) -> u[1]^2)
continuous_observed = ContinuousMeasurement(ControlParameter(0.:1.:2., name = :w2), (u, p, t) -> p[1] + u[1]^2)
CorleoneOED.add_observed!(symbolic_system, discrete_observed, continuous_observed)
println("  Added $(length(symbolic_system.discrete_measurements)) discrete and $(length(symbolic_system.continuous_measurements)) continuous measurements")
println("  Fisher variables: $(length(symbolic_system.fisher_continuous_vars)) elements")

println("\nStep 4: Creating new layer...")
new_layer = SingleShootingLayer(symbolic_system, layer)
new_prob = Corleone.get_problem(new_layer)
println("  New problem has $(length(new_prob.u0)) states (original: $(length(prob.u0)))")

println("\nStep 5: Creating OED layer...")
oed_layer = OEDLayerV2(symbolic_system, new_layer)

println("\nStep 6: Solving the augmented system...")
rng = Random.default_rng()
ps = LuxCore.initialparameters(rng, oed_layer)
st = LuxCore.initialstates(rng, oed_layer)
(fisher, traj), st_new = oed_layer(nothing, ps, st)
println("  Solution computed with $(length(traj.t)) time points")
println("  Final state size: $(length(traj.u[end]))")
println("  Fisher information from OED layer: $fisher")

println("\nStep 7: Extracting Fisher information separately...")
F = CorleoneOED.fisher_information(oed_layer, traj)
println("  Fisher information matrix:")
println("    F = $F")
println("  (Should match the Fisher from Step 6)")

println("\nStep 8: Extracting sensitivities...")
sens = CorleoneOED.sensitivities(oed_layer, traj)
println("  Sensitivity at t=0: $(sens[1])")
println("  Sensitivity at t=end: $(sens[end])")

println("\n✓ Example completed successfully!")
