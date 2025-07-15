using Corleone
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using OrdinaryDiffEq
using CairoMakie
using SciMLSensitivity, Optimization, OptimizationMOI, Ipopt
using SciMLSensitivity.ForwardDiff, SciMLSensitivity.Zygote, SciMLSensitivity.ReverseDiff
#using blockSQP

# Lotka
@variables begin
    x(..) = 1.e-2, [tunable=false]
    y(..) = 0, [tunable = false]
    u(..) = 0, [input = true, bounds = (0, 1)]
end

∫ = Symbolics.Integral(t in (0., 1.))

@named lotka_volterra = System(
    [
        D(x(t)) ~ y(t),
        D(y(t)) ~ 1 - 2 * u(t)
    ], t, [x(t), y(t), u(t)], [];
    constraints=[x(0) - 1.e-2 ~ 0, y(0) ~ 0, x(1) - 1.e-2 ~ 0, y(1) ~ 0],
    costs=Num[∫((x(t))^2) * 100],
    consolidate=(x...)->first(x)[1], # Hacky, IDK what this is at the moment
)
N = 20
shooting_points = collect(LinRange(0., 1., 11))

grid = ShootingGrid(shooting_points)
controlmethod = DirectControlCallback(
    u(t) => (; timepoints=collect(LinRange(0.,  N / (N+1), N)),
        defaults= ones(N)
    )
)
builder = OCProblemBuilder(
    lotka_volterra, controlmethod, grid, ConstantInitialization(Dict(x(t) => 0.0, y(t) => 0.0))
)

# Instantiates the problem fully
builder = builder()

optfun = OptimizationProblem{true}(builder, AutoForwardDiff(), Tsit5())

# Initial plot
sol = optfun.f.f.predictor(optfun.u0, saveat = 0.01)[1];
# plot(sol, idxs = [:x, :y, :u])

callback(x,l) = begin
    sol = optfun.f.f.predictor(x.u, saveat = 0.1)[1];
    display(plot(sol, idxs = [:x, :y, :u]))
    return false
end

# Optimize
# sol = solve(optfun, BlockSQPOpt(); maxiters = 100, opttol = 1e-6, callback=callback,
#        options=blockSQP.sparse_options(), sparsity=optfun.f.f.predictor.permutation.blocks)

sol = solve(optfun, Ipopt.Optimizer(); max_iter = 100, tol = 1e-6,
            hessian_approximation="limited-memory", )

# Result plot
pred = optfun.f.f.predictor(sol.u, saveat = 0.01)[1];
f = Figure();
ax = Axis(f[1,1]);
plot!(ax, pred, idxs = [:x, :y, :u]);
f
