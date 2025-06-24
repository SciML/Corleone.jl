using Test
using Corleone
using Corleone.ModelingToolkit
using Corleone.ModelingToolkit: t_nounits as t, D_nounits as D

@mtkmodel LQR begin
@description "Linear quadratic regulator"
@variables begin
x(t)=1.0, [description = "State variable", tunable = false]
u(t)=0.0, [description = "Control variable", input = true]
end
@parameters begin
a=-1.0, [description = "Decay", tunable = false]
b=1.0, [description = "Input scale", tunable = false]
end
@equations begin
D(x) ~ a * x + b * u
end
@costs begin
Symbolics.Integral(t in (0., 10.)).(10.0*(x-3.0)^2 + 0.1*u^2)
end
@consolidate begin
(system_costs...) -> first(system_costs)[1]
end
end;

lqr_model = LQR(; name = :LQR)

u = ModelingToolkit.getvar(lqr_model, :u, namespace = false)

control = DirectControlCallback(Num(u) => (; timepoints = collect(0.0:0.5:9.5)));

problem = OCProblemBuilder(lqr_model, control, ShootingGrid([0.,]), DefaultsInitialization())

expandedproblem = problem()

using Optimization, OptimizationMOI, Ipopt
using OrdinaryDiffEqTsit5, SciMLSensitivity

optimization_problem = OptimizationProblem{true}(expandedproblem, AutoForwardDiff(), Tsit5())

optimization_solution = solve(optimization_problem, Ipopt.Optimizer(),)

@test isapprox(optimization_solution.u, [1.0, 6.865341837123803, 2.303020511034252, 3.084624489682982, 2.9506996290744407, 2.973658328965578, 2.969720781097132, 2.970395856955674, 2.9702800858740277, 2.9702999379278796, 2.9702965162164032, 2.9702971776852873, 2.9702967089587893, 2.970298802173118, 2.970286737658208, 2.9703568817904946, 2.969948951800308, 2.9723213263356705, 2.9585244447063057, 3.0387621686432014, 2.5721283077786925])
@test SciMLBase.ReturnCode.Success == optimization_solution.retcode
@test isapprox(optimization_solution.objective, 1.5772524e+01)
@test optimization_solution.stats.iterations == 3 