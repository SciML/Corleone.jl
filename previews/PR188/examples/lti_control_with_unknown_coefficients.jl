using Corleone, ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using OrdinaryDiffEqTsit5
using ComponentArrays, ForwardDiff, Optimization, OptimizationMOI, Ipopt
using LuxCore, Random, LinearAlgebra

@variables begin
    x₁(t) = 0.0, [tunable = false]
    x₂(t) = 0.0, [tunable = false]
    x₃(t) = 0.0, [tunable = false]
    u(t) = 0.0, [input = true, bounds = (-3.0, 3.0)]
end
@parameters begin
    a[1:2] = [-0.2, -0.1], [tunable = true, bounds = (-0.5, -0.05)]
    b₁ = 0.5, [tunable = true, bounds = (0.1, 1.0)]
end
@constants begin
    x_t[1:3] = [3.0, 1.0, -2.0]
end


A = [0.0 1.0 0.0; a[1] 0.0 1.0; 0.0 0.0 a[2]]
b = [b₁, 0.0, 1.0]
x = [x₁, x₂, x₃]
equations = D.(x) .~ A * x + b * u
t_inf = 10.0
terminal_constraint = EvalAt(t_inf).(x) .~ scalarize(x_t)
energy = Symbolics.Integral(t in (0.0, t_inf))(0.5 * u^2)

@named lti = System(equations, t; costs = [energy], constraints = terminal_constraint)

step = 0.5
starts = collect(0.0:step:(t_inf - step))
dynopt = CorleoneDynamicOptProblem(
    scalarize(lti), [], u => starts;
    algorithm = Tsit5(), tspan = (0.0, t_inf),
    shooting = collect(2.0:2.0:8.0), rng = MersenneTwister(42),
    problem_kwargs = (; abstol = 1.0e-10, reltol = 1.0e-10),
)

A_nominal = [0.0 1.0 0.0; -0.2 0.0 1.0; 0.0 0.0 -0.1]
b_nominal = [0.5, 0.0, 1.0]
@assert rank(hcat(b_nominal, A_nominal * b_nominal, A_nominal^2 * b_nominal)) == 3
transition = exp([A_nominal b_nominal; zeros(1, 4)] * step)
Φ, Γ = transition[1:3, 1:3], transition[1:3, 4]
N = length(starts)
reachability = hcat([Φ^(N - k) * Γ for k in 1:N]...)
control_guess = reachability \ ModelingToolkit.getdefault(x_t)
@assert maximum(abs, control_guess) < 3.0
input_name = Symbol(u)
dynopt.ps.parameter.parameters[input_name].parameter.ps .= control_guess
forward_initialization(MersenneTwister(42), dynopt.layer, dynopt.ps, dynopt.st)
initial_energy = optimization_objective(dynopt)
@assert maximum(abs, optimization_constraints(dynopt)) < 1.0e-7
(; initial_energy, max_initial_violation = maximum(abs, optimization_constraints(dynopt)))

optprob = OptimizationProblem(dynopt, AutoForwardDiff())
result = solve(
    optprob, Ipopt.Optimizer(); max_iter = 500, tol = 1.0e-7,
    hessian_approximation = "limited-memory", bound_relax_factor = 0.0, print_level = 0,
)
@assert successful_retcode(result)
_, reconstruct = pack(default_vectorizer(), dynopt.ps)
optimized = reconstruct(result.u)
trajectory, _ = dynopt.layer(nothing, optimized, optprob.p)
terminal_state = [last(trajectory[state]) for state in x]
coefficients = (;
    A₂₁ = last(trajectory[a[1]]), A₃₃ = last(trajectory[a[2]]), b₁ = last(trajectory[b₁]),
)
@assert isapprox(terminal_state, ModelingToolkit.getdefault(x_t); atol = 1.0e-5, rtol = 0.0)
@assert maximum(abs, optimization_constraints(dynopt, optimized, optprob.p)) < 1.0e-5
@assert result.objective <= initial_energy + 1.0e-6

A_fitted = [0.0 1.0 0.0; coefficients.A₂₁ 0.0 1.0; 0.0 0.0 coefficients.A₃₃]
b_fitted = [coefficients.b₁, 0.0, 1.0]
fitted_transition = exp([A_fitted b_fitted; zeros(1, 4)] * step)
Φ_fitted, Γ_fitted = fitted_transition[1:3, 1:3], fitted_transition[1:3, 4]
replayed_terminal = foldl(
    (state, input) -> Φ_fitted * state + Γ_fitted * input,
    trajectory[u]; init = zeros(3)
)
@assert isapprox(
    replayed_terminal, ModelingToolkit.getdefault(x_t); atol = 1.0e-5, rtol = 0.0
)
(; retcode = result.retcode, coefficients, terminal_state, energy = result.objective)

using CairoMakie
trajectoryplot(
    trajectory; skips = quadrature_symbols(dynopt.layer.sys),
    state_string = "x", control_string = "u", show_variable_residuals = false
)

        using Pkg
        Pkg.status()

        using InteractiveUtils
        InteractiveUtils.versioninfo()

        using Pkg
        Pkg.status(; mode = PKGMODE_MANIFEST)
