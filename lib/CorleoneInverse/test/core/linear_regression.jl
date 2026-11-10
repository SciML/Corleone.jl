using Test, CorleoneInverse, OrdinaryDiffEqTsit5, ComponentArrays, NonlinearSolve, LuxCore, StableRNGs

# u(t) = u₀ + a t is linear in (u₀, a), so the fit and its covariance are ordinary least squares.
problem = ODEProblem((u, p, t) -> [p[1]], [0.0], (0.0, 4.0), [0.5]; abstol = 1.0e-12, reltol = 1.0e-12)
layer = ShootingLayer(
    problem, Tsit5(),
    Tunable(1; kind = Val(:parameter)) => nothing, Tunable(1; kind = Val(:variable)) => [0.0]
)
t = collect(0.0:0.5:4.0)
noise = [0.3, -0.5, 0.2, 0.1, -0.4, 0.6, -0.2, 0.0, 0.1] ./ 10
y = 1.0 .+ 2.0 .* t .+ noise
X = [ones(length(t)) t]
β = X \ y
s² = sum(abs2, y - X * β) / (length(t) - 2)
ps, st = LuxCore.setup(StableRNG(1), layer)

@testset "Unweighted fit matches ordinary least squares" begin
    experiment = Experiment(:u₁ => (t, y))
    sol = solve(NonlinearLeastSquaresProblem(layer, experiment, ps, st), LevenbergMarquardt())
    est = ParameterEstimate(layer, experiment, sol.u, ps, st)
    @test est.x ≈ β
    @test est.covariance ≈ s² * inv(X' * X)
    @test est.dof == length(t) - 2
    @test all(map((ci, x) -> first(ci) < x < last(ci), est.confidence_intervals, est.x))
    @test sprint(show, MIME"text/plain"(), est) isa String
end

@testset "Known noise scales do not rescale the covariance" begin
    experiment = Experiment(1 => (t, y, 0.1))
    sol = solve(NonlinearLeastSquaresProblem(layer, experiment, ps, st), LevenbergMarquardt())
    est = ParameterEstimate(layer, experiment, sol.u, ps, st)
    @test est.x ≈ β
    @test est.covariance ≈ 0.01 * inv(X' * X)
end

@testset "Experiment predictions sample every observation time" begin
    experiment = Experiment(:u₁ => (t[1:2:end], y[1:2:end]), ((u, p, t) -> u[1] + p[1]) => (t[2:2:end], y[2:2:end]))
    @test experiment.t == t
    @test n_observations(experiment) == length(t)
    state = experiment_state(layer, experiment, st)
    @test state.timestops == t
    traj, _ = layer(experiment, ps, state)
    @test traj.t == t
    @test residual((experiment,), (traj,)) == residual(experiment, traj)
    sol = solve(NonlinearLeastSquaresProblem(layer, experiment, ps, st), LevenbergMarquardt())
    # The second observation reads u₁ + a, a shifted design row.
    Xshift = copy(X)
    Xshift[2:2:end, 2] .+= 1
    @test sol.u ≈ Xshift \ y
end

@testset "Validation" begin
    @test_throws ArgumentError Experiment()
    @test_throws DimensionMismatch Experiment(:u₁ => (t, y[1:3]))
    @test_throws ArgumentError Experiment(:u₁ => (reverse(t), y))
    @test_throws ArgumentError Experiment(:u₁ => (t, y, -1.0))
    @test_throws DimensionMismatch Experiment(:u₁ => (t, y, [0.1, 0.2]))
    @test_throws ArgumentError experiment_state(layer, Experiment(:u₁ => ([5.0], [1.0])), st)
    @test_throws DimensionMismatch experiment_state(layer, Experiment(:u₁ => (t, y); u0 = [1.0, 2.0]), st)
    mixed = (Experiment(:u₁ => (t, y, 0.1)), Experiment(:u₁ => (t, y)))
    @test_throws ArgumentError NonlinearLeastSquaresProblem(layer, mixed, ps, st)
end
