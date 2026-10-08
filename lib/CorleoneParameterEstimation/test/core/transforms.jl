using Corleone, CorleoneParameterEstimation
using OrdinaryDiffEqTsit5
using NonlinearSolve
using LuxCore
using Test
using Random
using StableRNGs

rng = StableRNG(2024)

# Model: du/dt = -p[1]*u, exact solution u(t) = u0 * exp(-p[1]*t).
f = (du, u, p, t) -> (du[1] = -p[1] * u[1]; nothing)
p_true = 0.7
u0 = [1.0]
tspan = (0.0, 5.0)
prob = ODEProblem(f, u0, tspan, [0.3])

tdata = collect(0.0:0.25:5.0)
u_exact = u0[1] .* exp.(-p_true .* tdata)
σ = 0.01
ydata = u_exact .+ σ .* randn(rng, length(tdata))
data = ExperimentData(tdata, ydata; σ = σ)

# --------------------------------------------------------------------------------------
# Transform algebra
# --------------------------------------------------------------------------------------
@testset "Transform round trips and derivatives" begin
    transforms = [
        Identity(),
        Affine(2.5, -1.0),
        Log(),
        Logit(0.1, 3.0),
    ]
    θs = [0.7, 0.7, 0.7, 0.7]
    for (t, θ) in zip(transforms, θs)
        φ = CorleoneParameterEstimation.to_internal(t, θ)
        @test isapprox(CorleoneParameterEstimation.from_internal(t, φ), θ; rtol = 1.0e-12)
        # Central finite difference of `from_internal` matches the analytic derivative.
        h = 1.0e-6 * max(1.0, abs(φ))
        fd = (CorleoneParameterEstimation.from_internal(t, φ + h) -
              CorleoneParameterEstimation.from_internal(t, φ - h)) / (2h)
        @test isapprox(CorleoneParameterEstimation.dθ_dφ(t, φ), fd; rtol = 1.0e-5)
    end

    # Affine is exactly linear: the derivative is the constant scale.
    @test CorleoneParameterEstimation.dθ_dφ(Affine(2.5, -1.0), 12.3) == 2.5
    # Log/Logit are strictly positive-derivative bijections.
    @test CorleoneParameterEstimation.dθ_dφ(Log(), 0.7) > 0
    @test CorleoneParameterEstimation.dθ_dφ(Logit(0.1, 3.0), 0.7) > 0

    # Logit stays inside its interval for extreme internal values.
    @test CorleoneParameterEstimation.from_internal(Logit(0.1, 3.0), -1000.0) >= 0.1
    @test CorleoneParameterEstimation.from_internal(Logit(0.1, 3.0), 1000.0) <= 3.0

    # Invalid constructions are rejected.
    @test_throws ArgumentError Affine(0.0, 1.0)
    @test_throws ArgumentError Logit(2.0, 1.0)
end

# --------------------------------------------------------------------------------------
# Fitting with transforms
# --------------------------------------------------------------------------------------
@testset "Log transform fit is positive and matches the natural fit" begin
    pe = ParameterEstimationLayer(prob, Tsit5(), data)
    # Natural (identity) reference fit.
    sol0 = solve(problem(pe), GaussNewton(); abstol = 1.0e-8, reltol = 1.0e-8)
    est0 = estimate_parameters(pe, sol0)

    # The internal problem is unconstrained (the positivity is encoded in the transform).
    pe_problem = CorleoneParameterEstimation.CorleoneParameterEstimationProblem(
        pe; transforms = [Log()])
    @test !any(isfinite, pe_problem.lb)
    @test !any(isfinite, pe_problem.ub)

    solL = solve(NonlinearLeastSquaresProblem(pe_problem), GaussNewton();
        abstol = 1.0e-8, reltol = 1.0e-8)
    @test SciMLBase.successful_retcode(solL)
    estL = estimate_parameters(pe_problem, solL)

    # Same optimum (the reparameterization does not change the model), and the reported
    # values/uncertainties are in natural units.
    @test isapprox(estL.x[1], p_true; atol = 0.05)
    @test isapprox(estL.x[1], est0.x[1]; rtol = 1.0e-4)
    @test isapprox(estL.std_errors[1], est0.std_errors[1]; rtol = 1.0e-4)
    # Positivity holds for the point estimate and the confidence interval.
    lo, hi = estL.confidence_intervals[1]
    @test lo > 0
    @test lo <= p_true <= hi
    # The interval is formed by back-transforming the *endpoints* of the internal interval,
    # so on the log scale it is the symmetric `log(x) ± tcrit · se_φ`.
    tcrit = CorleoneParameterEstimation._t_quantile(estL.degrees_of_freedom, 0.975)
    seφ = estL.std_errors[1] / estL.x[1]          # se_φ = se_θ / (dθ/dφ), dθ/dφ = θ
    @test isapprox(log(lo), log(estL.x[1]) - tcrit * seφ; rtol = 1.0e-8)
    @test isapprox(log(hi), log(estL.x[1]) + tcrit * seφ; rtol = 1.0e-8)
end

@testset "Affine transform is exactly linear (symmetric CI)" begin
    pe = ParameterEstimationLayer(prob, Tsit5(), data)
    est0 = estimate_parameters(pe, solve(problem(pe), GaussNewton();
        abstol = 1.0e-8, reltol = 1.0e-8))

    pe_problem = CorleoneParameterEstimation.CorleoneParameterEstimationProblem(
        pe; transforms = [Affine(0.1, 5.0)])
    solA = solve(NonlinearLeastSquaresProblem(pe_problem), GaussNewton();
        abstol = 1.0e-8, reltol = 1.0e-8)
    estA = estimate_parameters(pe_problem, solA)

    # The optimum is unchanged...
    @test isapprox(estA.x[1], est0.x[1]; rtol = 1.0e-5)
    # ... and, because the Jacobian of an affine map is constant, the delta method is exact:
    # the back-transformed interval is exactly symmetric and equal to the natural interval
    # (an affine reparameterization does not change the natural uncertainty).
    lo, hi = estA.confidence_intervals[1]
    @test isapprox(estA.x[1] - lo, hi - estA.x[1]; rtol = 1.0e-10)
    lo0, hi0 = est0.confidence_intervals[1]
    @test isapprox(hi - lo, hi0 - lo0; rtol = 1.0e-4)
end

@testset "Logit transform enforces its interval" begin
    pe = ParameterEstimationLayer(prob, Tsit5(), data)
    lb, ub = 0.1, 1.1
    pe_problem = CorleoneParameterEstimation.CorleoneParameterEstimationProblem(
        pe; transforms = [Logit(lb, ub)])
    solLi = solve(NonlinearLeastSquaresProblem(pe_problem), GaussNewton();
        abstol = 1.0e-8, reltol = 1.0e-8)
    estLi = estimate_parameters(pe_problem, solLi)

    @test isapprox(estLi.x[1], p_true; atol = 0.05)
    @test lb < estLi.x[1] < ub
    lo, hi = estLi.confidence_intervals[1]
    @test lb <= lo <= hi <= ub
end

@testset "Transforms work with multiple shooting and several experiments" begin
    # Two experiments sharing the same model, fitted with a Log transform on the rate.
    data2 = ExperimentData(
        Experiment(tdata, u_exact .+ σ .* randn(rng, length(tdata)); σ = σ, u0 = u0),
        Experiment(tdata, u_exact .+ σ .* randn(rng, length(tdata)); σ = σ, u0 = u0),
    )
    shooting_points = collect(0.0:1.0:5.0)
    pe = ParameterEstimationLayer(prob, Tsit5(), data2, shooting_points;
        observed = (u, p, t) -> u[1:1])
    pe_problem = CorleoneParameterEstimation.CorleoneParameterEstimationProblem(
        pe; transforms = [Log()])
    # Only the shared rate is transformed; the local shooting states keep their bounds.
    @test pe_problem.transforms == (Log(),)
    sol = solve(NonlinearLeastSquaresProblem(pe_problem), LevenbergMarquardt();
        abstol = 1.0e-8, reltol = 1.0e-8, maxiters = 200)
    @test SciMLBase.successful_retcode(sol)
    est = estimate_parameters(pe_problem, sol)
    @test isapprox(est.x[1], p_true; atol = 0.05)
    @test est.confidence_intervals[1][1] > 0
end

@testset "Profile likelihood back-transforms the interval" begin
    pe = ParameterEstimationLayer(prob, Tsit5(), data)
    pe_problem = CorleoneParameterEstimation.CorleoneParameterEstimationProblem(
        pe; transforms = [Log()])
    sol = solve(NonlinearLeastSquaresProblem(pe_problem), GaussNewton();
        abstol = 1.0e-8, reltol = 1.0e-8)

    pl = profile_likelihood(pe_problem, sol; n_points = 41)
    lo, hi = pl.confidence_intervals[1]
    # Positive, finite and covering the truth; the curve values are natural (positive).
    @test lo > 0
    @test isfinite(lo) && isfinite(hi)
    @test lo <= p_true <= hi
    @test all(>(0), profiles(pl)[1].values)
    @test isapprox(pl.x[1], p_true; atol = 0.05)
    # Consistent with the asymptotic interval of the same (transformed) problem.
    alo, ahi = pl.asymptotic_intervals[1]
    @test isapprox(lo, alo; rtol = 0.3) || isapprox(lo, alo; atol = 0.02)
    @test isapprox(hi, ahi; rtol = 0.3) || isapprox(hi, ahi; atol = 0.02)
end

@testset "Transforms are type-stable for a homogeneous choice" begin
    pe = ParameterEstimationLayer(prob, Tsit5(), data)
    pe_problem = CorleoneParameterEstimation.CorleoneParameterEstimationProblem(
        pe; transforms = [Log()])
    @test isconcretetype(typeof(pe_problem))
    @test typeof(pe_problem.transforms) === Tuple{Log}
    # The residual closures still infer to a concrete `Vector{Float64}`.
    @test Base.return_types(
        getfield(pe_problem, :residual),
        (Vector{Float64}, typeof(pe_problem.x0), Nothing)
    )[1] === Vector{Float64}
    @test Base.return_types(
        getfield(pe_problem, :residual_oop),
        (typeof(pe_problem.x0), Nothing)
    )[1] === Vector{Float64}
end

@testset "Mixed transforms stay type-stable" begin
    # A tunable initial condition plus the model parameter, with different transforms. Storing
    # the transforms as a `Tuple` (rather than a `Vector{AbstractParameterTransform}`) keeps
    # the transform application and hence the residual closures concretely typed.
    tdata_m = collect(0.0:0.25:5.0)
    ydata_m = exp.(-p_true .* tdata_m) .+ σ .* randn(rng, length(tdata_m))
    data_m = ExperimentData(tdata_m, ydata_m; σ = σ)
    layer = SingleShootingLayer(prob, Tsit5(); saveat = unique(tdata_m),
        tunable_ic = [1], bounds_ic = ([0.0], [5.0]))
    pe = ParameterEstimationLayer(layer, data_m)
    pe_problem = CorleoneParameterEstimation.CorleoneParameterEstimationProblem(
        pe; transforms = [Identity(), Log()])
    @test isconcretetype(typeof(pe_problem))
    @test typeof(pe_problem.transforms) === Tuple{Identity, Log}
    @test pe_problem.x0 isa Vector{Float64}
    @test Base.return_types(
        getfield(pe_problem, :residual),
        (Vector{Float64}, typeof(pe_problem.x0), Nothing)
    )[1] === Vector{Float64}
    @test Base.return_types(
        getfield(pe_problem, :residual_oop),
        (typeof(pe_problem.x0), Nothing)
    )[1] === Vector{Float64}
end

@testset "Transform validation" begin
    pe = ParameterEstimationLayer(prob, Tsit5(), data)
    @test_throws ArgumentError CorleoneParameterEstimation.CorleoneParameterEstimationProblem(
        pe; transforms = [Log(), Log()])   # wrong length (only one shared variable)
    @test_throws ArgumentError CorleoneParameterEstimation.CorleoneParameterEstimationProblem(
        pe; transforms = [1.0])            # not a transform
    # The default start (natural p = 0.3) is outside a `Logit(0.4, 1.1)` interval; this is
    # reported as a clear error rather than an obscure `DomainError`.
    @test_throws ArgumentError CorleoneParameterEstimation.CorleoneParameterEstimationProblem(
        pe; transforms = [Logit(0.4, 1.1)])
end
