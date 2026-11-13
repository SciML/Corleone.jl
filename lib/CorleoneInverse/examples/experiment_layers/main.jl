# # Experiment-local annotations
#
# Two experiments observe the same first-order process
#
# ```math
# \dot{u} = -k u + v, \qquad p = [k, v],
# ```
#
# with a shared, unknown decay rate ``k``. Each experiment has its own unknown
# initial value and its own *known* input ``v``, given as a `Constant` annotation.
using CorleoneInverse, LuxCore, Random, ComponentArrays, NonlinearSolve, OrdinaryDiffEqTsit5
using TransformVariables: asℝ₊

function decay!(du, u, p, t)
    k, v = p
    du[1] = -k * u[1] + v
    return nothing
end
problem = ODEProblem(
    decay!, [1.0], (0.0, 3.0), [0.5, 0.0]; abstol = 1.0e-10, reltol = 1.0e-10
)

# The analytic solution generates noise-free data with ``k = 0.7``.
t = collect(0.1:0.1:3.0)
truth(t, u0, input, rate) = (u0 - input / rate) * exp(-rate * t) + input / rate

# ## Shared and local annotations
# The first annotations after the problem are shared by all experiments. Each
# experiment adds its own annotations, which override shared defaults for the same
# entry. Here every experiment owns a local initial value and a fixed input.
experiments = Experiments(
    problem,
    Tunable(1; transform = asℝ₊) => nothing,
    Tunable(1; kind = Val(:variable), transform = asℝ₊) => [0.0];
    controls = [2], experiments = (
        baseline = (
            annotations = (
                Tunable(1; kind = Val(:variable), transform = asℝ₊) => [0.0],
                Constant(2) => nothing,
            ),
            observations = (1 => (t, truth.(t, 1.0, 0.0, 0.7), 0.02),),
        ),
        forced = (
            annotations = (
                Tunable(1; kind = Val(:variable), default = 1.5, transform = asℝ₊) => [0.0],
                Constant(2; default = 0.3) => nothing,
            ),
            observations = (1 => (t, truth.(t, 2.0, 0.3, 0.7), 0.02),),
        ),
    )
)
shooting = ShootingLayer(problem, Tsit5())
ps, st = LuxCore.setup(Xoshiro(1), experiments)

# Calling the experiments returns one `(parameters, state)` pair per experiment,
# ready to be passed to the shooting layer.
calls, _ = experiments(nothing, ps, st)
@assert length(calls) == 2

# ## Fit
# The unknowns are the shared rate and the two local initial values.
model = CorleoneDynamicOptProblem(shooting, experiments; ps, st)
fit = solve(NonlinearLeastSquaresProblem(model), LevenbergMarquardt())
estimate = ParameterEstimate(model, fit.u)
@assert isapprox(estimate.x, [0.7, 1.0, 2.0]; rtol = 1.0e-5)

# Simulating all experiments at once gives the same result as simulating a single
# experiment with its own parameters and state.
_, reconstruct = pack(default_vectorizer(), ps)
fitted = reconstruct(fit.u)
trajectories, next_st = shooting(experiments, fitted, st)
fitted_calls, _ = experiments(nothing, fitted, st)
direct, _ = shooting(nothing, fitted_calls[2]...)
@assert direct.u ≈ trajectories[2].u
@assert trajectories isa Vector
@assert first(trajectories[2].u) ≈ [2.0]
estimate
