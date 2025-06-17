using Corleone
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using OrdinaryDiffEq
using CairoMakie
using SciMLSensitivity, Optimization, OptimizationMOI, Ipopt
using SciMLSensitivity.ForwardDiff, SciMLSensitivity.Zygote, SciMLSensitivity.ReverseDiff
using blockSQP

# Lotka
@variables begin
    x(..) = 0.5, [tunable=false]
    y(..) = 0.7, [tunable = false]
    u(..)=0.0 , [input = true, bounds = (0., 1.)]
end
@parameters begin
    p[1:4] = [1.0; 0.4; 1.0; 0.2], [tunable = false]
end
∫ = Symbolics.Integral(t in (0., 12.))

@named lotka_volterra = System(
    [
        D(x(t)) ~ x(t) - p[1] * x(t) * y(t) - p[2] * x(t) * u(t),
        D(y(t)) ~ -y(t) + p[3] * x(t) * y(t) - p[4] * y(t) * u(t)
    ], t, [x(t), y(t), u(t)], [p];
    costs=Num[∫((x(t) - 1)^2 + (y(t) - 1)^2)],
    consolidate=(x...)->first(x)[1], # Hacky, IDK what this is at the moment
)
N = 24
shooting_points = [0.,3., 6., 9.]

grid = ShootingGrid(shooting_points)
controlmethod = DirectControlCallback(
    u(t) => (; timepoints=collect(12 .* LinRange(0.,  N / (N+1), N)),
        defaults= collect(LinRange(0., 1., N))
    )
)
builder = OCProblemBuilder(
    lotka_volterra, controlmethod, grid, ForwardSolveInitialization()
)

# Instantiates the problem fully
builder = builder()

optfun = OptimizationProblem{true}(builder, AutoForwardDiff(), Tsit5())

# Initial plot
sol = optfun.f.f.predictor(optfun.u0, saveat = 0.01)[1];
plot(sol, idxs = [:x, :y, :u])

# Optimize
sol = solve(optfun, BlockSQPOpt(); maxiters = 100, opttol = 1e-6,
        options=blockSQP.sparse_options(), sparsity=optfun.f.f.predictor.permutation.blocks)

sol = solve(optfun, Ipopt.Optimizer(); max_iter = 100, tol = 1e-6,
         hessian_approximation="limited-memory", )

# Result plot
pred = optfun.f.f.predictor(sol.u, saveat = 0.01)[1]
f = plot(pred, idxs = [:x, :y, :u])