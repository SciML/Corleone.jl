using CorleoneInverse, LuxCore, Random, ComponentArrays, NonlinearSolve, OrdinaryDiffEqTsit5
using TransformVariables: asℝ₊

problem = ODEProblem(
    (u, p, t) -> [-p[1] * u[1] + p[2]], [1.0], (0.0, 3.0), [0.5, 0.0];
    abstol = 1.0e-10, reltol = 1.0e-10
)
t = collect(0.1:0.1:3.0)
truth(t, u0, input, rate) = (u0 - input / rate) * exp(-rate * t) + input / rate
# Both experiments share the decay parameter. Each supplies a local initial value
# and a fixed control.
experiments = Experiments(
    problem,
    Tunable(1; transform = asℝ₊) => nothing,
    Tunable(1; kind = Val(:variable), transform = asℝ₊) => [0.0];
    controls = [2], experiments = (
        baseline = (
            annotations = (Tunable(1; kind = Val(:variable), transform = asℝ₊) => [0.0], Constant(2) => nothing),
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
calls, _ = experiments(nothing, ps, st)
@assert length(calls) == 2
model = CorleoneDynamicOptProblem(shooting, experiments; ps, st)
fit = solve(NonlinearLeastSquaresProblem(model), LevenbergMarquardt())
estimate = ParameterEstimate(model, fit.u)
@assert isapprox(estimate.x, [0.7, 1.0, 2.0]; rtol = 1.0e-5)
_, reconstruct = pack(default_vectorizer(), ps)
fitted = reconstruct(fit.u)
trajectories, next_st = shooting(experiments, fitted, st)
fitted_calls, _ = experiments(nothing, fitted, st)
direct, _ = shooting(nothing, fitted_calls[2]...)
@assert direct.u ≈ trajectories[2].u
@assert trajectories isa Vector
@assert first(trajectories[2].u) ≈ [2.0]
estimate
