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
    x(..) = 0., [tunable=false, bounds = (-0.25, Inf)]
    y(..) = 1., [tunable = false]
    u(..) = 0.0, [input = true, bounds = (0., 1.)]
end
@parameters begin
    p[1:2] = [1, 1], [tunable = false]
end

∫ = Symbolics.Integral(t in (0., 10.))

@named lotka_volterra = System(
    [
        D(x(t)) ~ (1 - y(t)^2) * x(t) - y(t) + u(t)
        D(y(t)) ~ x(t) * p[1]
    ], t, [x(t), y(t), u(t)], [p];
    constraints=reduce(vcat, [(-x(ti) -0.25 ≲ 0.0) for ti in collect(0.0:0.05:3.0)]),
    costs=Num[∫((x(t))^2 + (y(t))^2 + (u(t))^2)],
    consolidate=(x...)->first(x)[1], # Hacky, IDK what this is at the moment
)
N = 20
shooting_points = [0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]

grid = ShootingGrid(shooting_points)
controlmethod = DirectControlCallback(
    u(t) => (; timepoints=collect(10 .* LinRange(0.,  N / (N+1), N)),
        defaults= zeros(N)
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
plot(sol, idxs = [:x, :y, :u])

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
ax = Axis(f[1,1], xticks=0:1:10, yticks=-0.5:0.25:1.0, limits=(nothing, (-0.5, 1)));
plot!(ax, pred, idxs = [:x, :y, :u]);
f
