using DifferentiationInterface: AutoFiniteDiff, AutoForwardDiff, AutoReverseDiff, check_available
using FiniteDiff
using ForwardDiff
using ReverseDiff
using SciMLSensitivity

include("suite.jl")
using .SequentialADTests: test_sequential_ad

test_sequential_ad(AutoForwardDiff())
@test check_available(AutoReverseDiff())
# Keep full solutions (including transition metadata) while ReverseDiff traces
# the inner solver; the default adjoint rule returns only a tracked state array.
test_sequential_ad(
    AutoReverseDiff();
    solve_kwargs = (; sensealg = SciMLSensitivity.SensitivityADPassThrough())
)
# Numerical differentiation is a baseline, not an AD implementation.
test_sequential_ad(AutoFiniteDiff(); gradient_atol = 1.0e-5, gradient_rtol = 1.0e-5)
