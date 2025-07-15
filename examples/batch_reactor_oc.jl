using Corleone
using TestEnv
# TestEnv.activate()
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using OrdinaryDiffEqTsit5
using CairoMakie
using SciMLSensitivity, Optimization, OptimizationMOI, Ipopt
using SciMLSensitivity.ForwardDiff, SciMLSensitivity.Zygote, SciMLSensitivity.ReverseDiff
#using blockSQP

# Batch Reactor
@variables begin
    x(..) = 1., [tunable=false]
    y(..) = 0., [tunable = false]
    u(..) = 300.0, [input = true, bounds = (298., 398.)]
end

# auxiliary equations
#k1 = 4.e3 * exp(-2500 / u(t))
#k2 = 62.e4 * exp(-5000 / u(t))

∫ = Symbolics.Integral(t in (0., 1.))

@named batch_reactor = System(
    [
        D(x(t)) ~ -4.e3 * exp(-2500 / u(t)) * x(t)^2
        D(y(t)) ~ 4.e3 * exp(-2500 / u(t)) * x(t)^2 - 62.e4 * exp(-5000 / u(t)) * y(t)^2
    ], t, [x(t), y(t), u(t)], [];
    # constraints=reduce(vcat, [(-x(ti) -0.25 ≲ 0.0) for ti in collect(0.0:0.05:3.0)]),
    costs=[-y(1.)],
    consolidate=(x...)->first(x)[1], # Hacky, IDK what this is at the moment
)
N = 20
shooting_points = [0., 0.5, 1.]

grid = ShootingGrid(shooting_points)
controlmethod = DirectControlCallback(
    u(t) => (; timepoints=collect(1 .* LinRange(0.,  N / (N+1), N)),
        defaults= 300 * ones(N)
    )
)
builder = OCProblemBuilder(
    batch_reactor, controlmethod, grid, ConstantInitialization(Dict(x(t) => 1.0, y(t) => 0.0))
)

# Instantiates the problem fully

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
