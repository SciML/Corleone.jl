using Corleone, CorleoneParameterEstimation
using OrdinaryDiffEqTsit5
using NonlinearSolve
using LuxCore
using Test
using Random
using StableRNGs

rng = StableRNG(42)

# Model: du/dt = -p[1]*u, exact solution u(t) = u0 * exp(-p[1]*t).
f = (du, u, p, t) -> (du[1] = -p[1] * u[1]; nothing)
p_true = 0.7
u0 = [1.0]
tspan = (0.0, 5.0)
prob = ODEProblem(f, u0, tspan, [0.3])

tdata = collect(0.0:0.2:5.0)
u_exact = u0[1] .* exp.(-p_true .* tdata)
σ = 0.01
ydata = u_exact .+ σ .* randn(rng, length(tdata))
data = ExperimentData(tdata, ydata; σ = σ)

@testset "Profile likelihood recovers parameter and matches asymptotic CI" begin
    pe = ParameterEstimationLayer(prob, Tsit5(), data)
    nlls = problem(pe)
    sol = solve(nlls, GaussNewton(); abstol = 1e-8, reltol = 1e-8)
    @test SciMLBase.successful_retcode(sol)

    est = profile_likelihood(pe, sol; names = [:p])
    @test est isa ProfileLikelihoodEstimate
    @test est.n_parameters == 1
    @test length(est.profiles) == 1

    # The point estimate is reproduced.
    @test isapprox(est.x[1], p_true; atol = 0.05)

    # The profile interval covers the truth...
    lo, hi = est.confidence_intervals[1]
    @test lo <= p_true + 1.0e-6
    @test p_true - 1.0e-6 <= hi

    # ... and is consistent with the (linearization) asymptotic interval for this
    # near-linear problem, so the two cross-checks agree.
    alo, ahi = est.asymptotic_intervals[1]
    @test isfinite(alo) && isfinite(ahi)
    @test isapprox(lo, alo; rtol = 0.2) || isapprox(lo, alo; atol = 0.02)
    @test isapprox(hi, ahi; rtol = 0.2) || isapprox(hi, ahi; atol = 0.02)

    # Profile is a proper valley: RSS is minimized near the point estimate and a threshold
    # was crossed at both ends.
    curve = est.profiles[1]
    @test length(curve.values) == length(curve.rss)
    @test curve.min_rss ≈ minimum(curve.rss) atol = 1.0e-6
    @test curve.threshold > curve.min_rss
    @test curve.lower < est.x[1] < curve.upper

    # Accessors agree with the struct fields.
    @test confidence_interval(est)[1] == (lo, hi)
    @test profiles(est)[1] === est.profiles[1]
end

@testset "Profile interval cross-checks the asymptotic interval" begin
    # The profile follows the true nonlinear likelihood while the asymptotic one is a
    # first-order (linearization) approximation, so for a mildly nonlinear problem the two
    # must agree to within a modest factor -- the whole point of using profile likelihood as
    # a cross-check. We assert agreement to within 2.5x rather than an exact ordering to
    # stay robust to the (small) curvature of the likelihood.
    pe = ParameterEstimationLayer(prob, Tsit5(), data)
    sol = solve(problem(pe), GaussNewton(); abstol = 1e-8, reltol = 1e-8)
    est = profile_likelihood(pe, sol; n_points = 61)
    lo, hi = est.confidence_intervals[1]
    alo, ahi = est.asymptotic_intervals[1]
    half = (hi - lo) / 2
    ahalf = (ahi - alo) / 2
    @test ahalf / 2.5 <= half <= ahalf * 2.5 + 1.0e-6
end

@testset "Flat (non-identifiable) parameter yields an unbounded interval" begin
    # Add a second parameter that does not enter the RHS at all (only p[1] is used). It is
    # therefore completely unidentifiable and its profile is flat: RSS never rises above the
    # threshold, so the interval must be reported unbounded.
    f2 = (du, u, p, t) -> (du[1] = -p[1] * u[1]; nothing)
    prob2 = ODEProblem(f2, u0, tspan, [0.3, 99.0])
    data2 = ExperimentData(tdata, ydata; σ = σ)
    pe = ParameterEstimationLayer(prob2, Tsit5(), data2)
    sol = solve(problem(pe), GaussNewton(); abstol = 1e-8, reltol = 1e-8)

    est = profile_likelihood(pe, sol; names = [:alpha, :dummy], n_points = 41)
    # The identifiable parameter attains a bordered interval.
    @test est.confidence_intervals[1][1] != -Inf
    @test est.confidence_intervals[1][2] != Inf
    # The dummy parameter is not identified -> unbounded both sides.
    @test est.confidence_intervals[2][1] == -Inf
    @test est.confidence_intervals[2][2] == Inf
end

@testset "Internal chi-square quantile and crossing helpers" begin
    # χ²_1 at the 95% level is 3.8415.
    @test isapprox(CorleoneParameterEstimation._chi_square_quantile(1, 0.95), 3.8415; atol = 1.0e-3)
    # χ²_1 at the 99% level is 6.6349.
    @test isapprox(CorleoneParameterEstimation._chi_square_quantile(1, 0.99), 6.6349; atol = 1.0e-2)

    # Linear crossing: RSS crosses threshold 5 exactly midway between (x=2, rss=6) and
    # (x=4, rss=4).
    cr = CorleoneParameterEstimation._interp_cross(2.0, 6.0, 4.0, 4.0, 5.0)
    @test isapprox(cr, 3.0; atol = 1.0e-12)

    # Endpoint reading on a synthetic V-shaped profile.
    vals = collect(range(0.0, 4.0, length = 9))   # 0, 0.5, ..., 4
    rss = abs.(vals .- 2.0) .+ 1.0                # min at 2.0, rss=1
    lo, hi = CorleoneParameterEstimation._profile_ci_endpoints(vals, rss, 2.5, 2.0)
    # threshold 2.5 crossed at |v - 2| = 1.5 -> v in [0.5, 3.5]
    @test isapprox(lo, 0.5; atol = 1.0e-3)
    @test isapprox(hi, 3.5; atol = 1.0e-3)
end
