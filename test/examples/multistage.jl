using Corleone
using ComponentArrays
using Ipopt
using Lux
using LuxCore
using Optimization
using OptimizationMOI
using OrdinaryDiffEqTsit5
using Random
using SciMLBase
using Test

# Two-stage SIR problem with free stage durations and a parameterized transition. Both
# stages are formulated in normalized time and their right-hand sides are scaled by the
# stage durations `T₁` and `T₂`. The stages are glued together by a `StageTransition`
# whose instantaneous vaccination fraction `η` is optimized as well.

const N = 1.0
const β = 0.6
const γ = 0.2
const ν = 0.1
const a = 10.0
const w = 0.5
const c_switch = 0.5
const T_total = 10.0

function sir_lockdown!(du, u, p, t)
    S, I, R, C = u
    T1, u₁ = p
    infection = β * (1 - u₁) * S * I / N
    du[1] = T1 * (-infection)
    du[2] = T1 * (infection - γ * I)
    du[3] = T1 * (γ * I)
    du[4] = T1 * (a * I + w * u₁^2)
    return
end

function sir_vaccination!(du, u, p, t)
    S, I, R, C = u
    T2, u₂ = p
    infection = β * S * I / N
    vaccination = ν * u₂ * S
    du[1] = T2 * (-infection - vaccination)
    du[2] = T2 * (infection - γ * I)
    du[3] = T2 * (γ * I + vaccination)
    du[4] = T2 * (a * I + w * u₂^2)
    return
end

u0 = [0.99, 0.01, 0.0, 0.0]
prob_lockdown = ODEProblem(sir_lockdown!, u0, (0.0, 1.0), [T_total / 2, 0.0])
prob_vaccination = ODEProblem(sir_vaccination!, u0, (0.0, 1.0), [T_total / 2, 0.0])

grid = collect(0.0:0.1:0.9)

lockdown = ControlParameter(
    grid, name = :lockdown, bounds = (0.0, 1.0), controls = zeros(length(grid))
)
vaccination = ControlParameter(
    grid, name = :vaccination, bounds = (0.0, 1.0), controls = zeros(length(grid))
)

stage_lockdown = SingleShootingLayer(
    prob_lockdown, Tsit5();
    controls = (2 => lockdown,), bounds_p = ([0.1], [T_total - 0.1])
)
stage_vaccination = SingleShootingLayer(
    prob_vaccination, Tsit5();
    controls = (2 => vaccination,), bounds_p = ([0.1], [T_total - 0.1])
)

function vaccination_switch(traj, p)
    S, I, R, C = last(traj.u)
    η = p[1]
    jump = η * S
    return [S - jump, I, R + jump, C + c_switch * jump]
end

transition = StageTransition(vaccination_switch; p = [0.0], bounds = ([0.0], [0.3]))
chain = Chain(stage_lockdown, transition, stage_vaccination)

# Structure of the composed problem.
@test LuxCore.parameterlength(chain) ==
    LuxCore.parameterlength(stage_lockdown) + 1 + LuxCore.parameterlength(stage_vaccination)

lb, ub = Corleone.get_bounds(chain)
ps, st = LuxCore.setup(Random.default_rng(), chain)
p = ComponentArray(ps)
@test length(ComponentArray(lb)) == length(p)
@test length(ComponentArray(ub)) == length(p)
@test ComponentArray(lb).layer_2.p == [0.0]
@test ComponentArray(ub).layer_2.p == [0.3]
@test Corleone.get_number_of_shooting_constraints(chain) == 0

# The transition maps the final state of the first stage to the initial state of the
# second stage, including the instantaneous vaccination and the switching cost.
@test transition.f((u = [[0.9, 0.05, 0.05, 1.0]],), [0.2]) ≈
    [0.9 * (1 - 0.2), 0.05, 0.05 + 0.2 * 0.9, 1.0 + c_switch * 0.2 * 0.9]
traj_lockdown, _ = stage_lockdown(nothing, ps.layer_1, st.layer_1)
u0_next = transition.f(traj_lockdown, ps.layer_2.p)
S_end, I_end, R_end, C_end = last(traj_lockdown.u)
@test u0_next ≈ [S_end, I_end, R_end, C_end] # η = 0 in the initial parameters

# Without controls the chain reproduces the baseline cost.
baseline, _ = chain(nothing, ps, st)
@test last(baseline.u)[4] ≈ 8.7837 atol = 1.0e-3

# A parameter constraint couples the durations of the two stages.
duration_constraint = (ps -> [ps.layer_1.p[1] + ps.layer_3.p[1]]) =>
    (lb = [T_total], ub = [T_total])

optprob = OptimizationProblem(
    chain, AutoForwardDiff(), Val(:ComponentArrays);
    loss = :x₄, parameter_constraints = duration_constraint
)
@test length(optprob.u0) == LuxCore.parameterlength(chain)
@test optprob.lcons == [T_total]
@test optprob.ucons == [T_total]

sol = solve(
    optprob, Ipopt.Optimizer(), max_iter = 500, tol = 1.0e-8,
    hessian_approximation = "limited-memory", print_level = 0
)

@test SciMLBase.successful_retcode(sol)
@test isapprox(sol.objective, 2.5667, atol = 1.0e-3)

p_opt = sol.u .+ zero(p)

# The parameter constraint holds at the solution and the horizon is split non-trivially.
@test p_opt.layer_1.p[1] + p_opt.layer_3.p[1] ≈ T_total atol = 1.0e-6
@test 0.1 < p_opt.layer_1.p[1] < T_total - 0.1
@test isapprox(p_opt.layer_1.p[1], 3.3883, atol = 1.0e-2)

# The transition fraction is optimized and respects its bounds.
@test -1.0e-6 <= p_opt.layer_2.p[1] <= 0.3 + 1.0e-6
@test isapprox(p_opt.layer_2.p[1], 0.3, atol = 1.0e-4)

traj_vaccination, _ = chain(nothing, p_opt, st)
traj_lockdown_opt, _ = stage_lockdown(nothing, p_opt.layer_1, st.layer_1)

# The transition carries the first stage's cost into the second stage and adds the
# switching cost proportional to the jump in `R`.
@test last(traj_vaccination.u)[4] ≈ sol.objective
S_before = last(traj_lockdown_opt.u)[1]
@test first(traj_vaccination.u)[4] ≈
    last(traj_lockdown_opt.u)[4] + c_switch * p_opt.layer_2.p[1] * S_before
