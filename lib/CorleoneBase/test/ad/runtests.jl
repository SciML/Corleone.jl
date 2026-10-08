using DifferentiationInterface: AutoFiniteDiff, AutoForwardDiff, AutoReverseDiff, AutoZygote,
    AutoMooncake, AutoMooncakeForward, check_available
using FiniteDiff
using ForwardDiff
using Mooncake
using ReverseDiff
using SciMLSensitivity
using Zygote

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
@test check_available(AutoZygote())
test_sequential_ad(AutoZygote())
include("zygote_rules.jl")
@test check_available(AutoMooncake())
test_sequential_ad(AutoMooncake())
@test check_available(AutoMooncakeForward())
test_sequential_ad(AutoMooncakeForward())
# Numerical differentiation is a baseline, not an AD implementation.
test_sequential_ad(AutoFiniteDiff(); gradient_atol = 1.0e-5, gradient_rtol = 1.0e-5)
