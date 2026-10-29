#src ---
#src title: LTI Control with Unknown Coefficients
#src description: Optimize a scalar input and three constant coefficients under terminal state constraints
#src tags:
#src   - ModelingToolkit
#src   - Linear System
#src   - Optimal Control
#src icon: 🎛️
#src ---

# ## [LTI Control with Unknown Coefficients](@id mtk_lti)
# We jointly optimize a scalar input and three constant coefficients in
# ``\dot{x} = A x + b u``, with ``x(0)=0`` and
# ``x(10)=[3,1,-2]^T``. The objective is minimum control energy,
# ``\frac12\int_0^{10}u(t)^2\,dt``.

using Corleone, ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using OrdinaryDiffEqTsit5
using ComponentArrays, ForwardDiff, Optimization, OptimizationMOI, Ipopt
using LuxCore, Random, LinearAlgebra

# ## Model
# Only A[2,1], A[3,3], and b[1] are unknown. All other entries are fixed.
# The physical initial state is fixed; interior shooting values remain tunable.
# The MTK array `a` contains the two unknown entries of A. Corleone scalarizes
# its components during setup and stores each in its own scalar vector buffer.

@variables begin
    x₁(..) = 0.0, [tunable = false]
    x₂(..) = 0.0, [tunable = false]
    x₃(..) = 0.0, [tunable = false]
    u(..) = 0.0, [input = true, bounds = (-3.0, 3.0)]
end
@parameters begin
    a[1:2] = [-0.2, -0.1], [tunable = true, bounds = (-0.5, -0.05)]
    b₁ = 0.5, [tunable = true, bounds = (0.1, 1.0)]
end

A = [0.0 1.0 0.0; a[1] 0.0 1.0; 0.0 0.0 a[2]]
b = [b₁, 0.0, 1.0]
x = [x₁(t), x₂(t), x₃(t)]
equations = D.(x) .~ A * x + b * u(t)
t_inf = 10.0
target = [3.0, 1.0, -2.0]
terminal = [x₁(t_inf), x₂(t_inf), x₃(t_inf)]
energy = Symbolics.Integral(t in (0.0, t_inf))(0.5 * u(t)^2)
@named lti = System(equations, t; costs = [energy], constraints = terminal .~ target)

# ## Shooting formulation
# Twenty piecewise-constant input intervals share one lookup. The three unknown
# coefficients have no switching grid, so they remain constant over the horizon
# and share another lookup. Each coefficient keeps its own scalar vector buffer.
# Every two time units we lift all states, including the energy quadrature.
# Corleone adds their continuity equalities to the three terminal equalities.

step = 0.5
starts = collect(0.0:step:(t_inf - step))
dynopt = CorleoneDynamicOptProblem(
    lti, [], u(t) => starts;
    algorithm = Tsit5(), tspan = (0.0, t_inf),
    shooting = collect(2.0:2.0:8.0), rng = MersenneTwister(42),
    problem_kwargs = (; abstol = 1.0e-10, reltol = 1.0e-10),
)

# ## Feasible initialization
# At the nominal coefficients, an exact zero-order-hold discretization gives
# ``x_{k+1}=\Phi x_k+\Gamma u_k``. Its reachability matrix supplies a minimum-norm
# input that reaches the target. This initializes the control without changing
# the layer or its parameter layout. Forward initialization fills the lifted
# states and quadrature from that input.

A_nominal = [0.0 1.0 0.0; -0.2 0.0 1.0; 0.0 0.0 -0.1]
b_nominal = [0.5, 0.0, 1.0]
@assert rank(hcat(b_nominal, A_nominal * b_nominal, A_nominal^2 * b_nominal)) == 3
transition = exp([A_nominal b_nominal; zeros(1, 4)] * step)
Φ, Γ = transition[1:3, 1:3], transition[1:3, 4]
N = length(starts)
reachability = hcat([Φ^(N - k) * Γ for k in 1:N]...)
control_guess = reachability \ target
@assert maximum(abs, control_guess) < 3.0
input_name = Symbol(u(t))
dynopt.ps.parameter.parameters[input_name].parameter.ps .= control_guess
forward_initialization(MersenneTwister(42), dynopt.layer, dynopt.ps, dynopt.st)
initial_energy = optimization_objective(dynopt)
@assert maximum(abs, optimization_constraints(dynopt)) < 1.0e-7

# ## Joint optimization
# The target alone does not identify unique coefficients. We treat them as design
# variables: their bounds and the energy objective define the joint optimization.
# We keep them shared between shooting segments; no parameter splitting is used.

optprob = OptimizationProblem(dynopt, AutoForwardDiff())
result = solve(
    optprob, Ipopt.Optimizer(); max_iter = 500, tol = 1.0e-7,
    hessian_approximation = "limited-memory", bound_relax_factor = 0.0, print_level = 0,
)
@assert Corleone.SciMLBase.successful_retcode(result)
optimized = optprob.f.f.reconstruct(result.u)
trajectory, _ = dynopt.layer(nothing, optimized, optprob.p)
terminal_state = [last(trajectory[state]) for state in x]
coefficients = (; A₂₁ = last(trajectory[a[1]]), A₃₃ = last(trajectory[a[2]]), b₁ = last(trajectory[b₁]))
@assert isapprox(terminal_state, target; atol = 1.0e-5, rtol = 0.0)
@assert maximum(abs, optimization_constraints(dynopt, optimized, optprob.p)) < 1.0e-5
@assert result.objective <= initial_energy + 1.0e-6
# Replaying the fitted constant system provides a check independent of lifted states.
A_fitted = [0.0 1.0 0.0; coefficients.A₂₁ 0.0 1.0; 0.0 0.0 coefficients.A₃₃]
b_fitted = [coefficients.b₁, 0.0, 1.0]
fitted_transition = exp([A_fitted b_fitted; zeros(1, 4)] * step)
Φ_fitted, Γ_fitted = fitted_transition[1:3, 1:3], fitted_transition[1:3, 4]
replayed_terminal = foldl(
    (state, input) -> Φ_fitted * state + Γ_fitted * input,
    trajectory[u(t)]; init = zeros(3)
)
@assert isapprox(replayed_terminal, target; atol = 1.0e-5, rtol = 0.0)
(; coefficients, terminal_state, energy = result.objective)

# ## Plot
# Plot the three physical states and the scalar input. The energy is reported above.
using CairoMakie
trajectoryplot(
    trajectory; skips = quadrature_symbols(dynopt.layer.sys),
    state_string = "x", control_string = "u", show_variable_residuals = false
)
