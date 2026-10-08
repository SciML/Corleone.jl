using DifferentiationInterface: AutoFiniteDiff, AutoForwardDiff
using FiniteDiff
using ForwardDiff

include("suite.jl")
using .SequentialADTests: test_sequential_ad

test_sequential_ad(AutoForwardDiff())
# Numerical differentiation is a baseline, not an AD implementation.
test_sequential_ad(AutoFiniteDiff(); gradient_atol = 1.0e-5, gradient_rtol = 1.0e-5)
