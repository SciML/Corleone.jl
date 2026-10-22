using Corleone, CorleoneParameterEstimation
using OrdinaryDiffEqTsit5
using NonlinearSolve
using LuxCore
using Test
using Random
using StableRNGs

# Infer the return type of `f` for the given argument types. This uses
# `Base.return_types` on the concrete function type rather than `@inferred` inside a
# zero-argument closure, which would report a spurious instability for non-`const`
# globals. Returns `Union{}` when no method matches and the `Union` of the per-method
# return types when several do.
function inferred_return_type(f, argtypes::Tuple)
    rts = Base.return_types(f, argtypes)
    isempty(rts) && return Union{}
    return length(rts) == 1 ? rts[1] : Union{rts...}
end

rng = StableRNG(1234)

# Model: du/dt = -p[1]*u, exact solution u(t) = u0 * exp(-p[1]*t).
f = (du, u, p, t) -> (du[1] = -p[1] * u[1]; nothing)
p_true = 0.7
prob = ODEProblem(f, [1.0], (0.0, 5.0), [0.3])
tdata = collect(0.0:0.25:5.0)
ydata = exp.(-p_true .* tdata) .+ 0.01 .* randn(rng, length(tdata))
data = ExperimentData(tdata, ydata; σ = 0.01)
shooting_points = collect(0.0:1.0:5.0)

@testset "single shooting" begin
    pe = ParameterEstimationLayer(prob, Tsit5(), data)
    pe_problem = CorleoneParameterEstimation.CorleoneParameterEstimationProblem(pe)
    P = typeof(pe_problem)

    # The problem struct is concrete and the residual counts are stored as concrete
    # `Int` fields. The layer's `initialstates` is not type-stable (it returns a
    # `Union` of `SymbolCache` types), so the counts are computed once at construction
    # and must not be recomputed downstream.
    @test isconcretetype(P)
    @test fieldtype(P, :n_residuals) === Int
    @test fieldtype(P, :n_shooting_constraints) === Int
    @test pe_problem.n_residuals isa Int
    @test pe_problem.n_shooting_constraints isa Int
    @test pe_problem.n_shooting_constraints == 0
    @test pe_problem.n_residuals == n_observations(data)

    # In-place residual closure `(res, x, p) -> res` and the out-of-place closure
    # `(x, p) -> r` both infer to a concrete `Vector{Float64}`.
    @test inferred_return_type(
        getfield(pe_problem, :residual),
        (Vector{Float64}, typeof(pe_problem.x0), Nothing)
    ) === Vector{Float64}
    @test inferred_return_type(
        getfield(pe_problem, :residual_oop),
        (typeof(pe_problem.x0), Nothing)
    ) === Vector{Float64}

    # The exported out-of-place residual (the reverse-mode-AD entry point) is stable.
    @test inferred_return_type(
        residual,
        (typeof(pe), typeof(pe_problem.ps0), typeof(pe_problem.st))
    ) === Vector{Float64}

    # `NonlinearLeastSquaresProblem` is built from the concrete out-of-place residual.
    @test typeof(problem(pe)) <: NonlinearLeastSquaresProblem

    # `estimate_parameters` infers a concretely-typed Jacobian, hence a concrete
    # covariance matrix (the ForwardDiff.jacobian result is type-asserted).
    sol = solve(problem(pe), GaussNewton(); abstol = 1.0e-6, reltol = 1.0e-6)
    @test SciMLBase.successful_retcode(sol)
    res = estimate_parameters(pe, sol)
    @test covariance(res) isa Matrix{Float64}
    @test standard_errors(res) isa Vector{Float64}
end

@testset "multiple shooting" begin
    pe = ParameterEstimationLayer(
        prob, Tsit5(), data, shooting_points;
        observed = (u, p, t) -> u[1:1]
    )
    pe_problem = CorleoneParameterEstimation.CorleoneParameterEstimationProblem(pe)
    P = typeof(pe_problem)

    @test isconcretetype(P)
    @test fieldtype(P, :n_residuals) === Int
    @test fieldtype(P, :n_shooting_constraints) === Int
    # 11 observations + 4 boundaries * (1 state + 1 parameter) continuity residuals.
    @test pe_problem.n_shooting_constraints == 4 * 2
    @test pe_problem.n_residuals == n_observations(data) + pe_problem.n_shooting_constraints

    @test inferred_return_type(
        getfield(pe_problem, :residual),
        (Vector{Float64}, typeof(pe_problem.x0), Nothing)
    ) === Vector{Float64}
    # The multiple-shooting layer's trajectory type is inferred as a `UnionAll`, but the
    # out-of-place residual asserts the concrete element type, so it stays stable.
    @test inferred_return_type(
        getfield(pe_problem, :residual_oop),
        (typeof(pe_problem.x0), Nothing)
    ) === Vector{Float64}
    @test inferred_return_type(
        residual,
        (typeof(pe), typeof(pe_problem.ps0), typeof(pe_problem.st))
    ) === Vector{Float64}
end
