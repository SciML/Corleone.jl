using Corleone
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using OrdinaryDiffEq
using SciMLSensitivity, Optimization, OptimizationMOI, Ipopt
using SciMLSensitivity.ForwardDiff, SciMLSensitivity.Zygote, SciMLSensitivity.ReverseDiff
using CairoMakie
using blockSQP

# Lotka
tspan = (0.0,12.0)
∫ = Symbolics.Integral(t in tspan)

@variables begin
    x(..) = 0.5, [tunable=false]
    y(..) = 0.7, [tunable = false]
    u(..)=0.0, [input = true, bounds = (0., 1.)]
    h1(..)=0.0, [tunable=false, bounds=(0.,1.)]
    h2(..)=0.0, [tunable=false, bounds=(0.,1.)]
end
@parameters begin
    p1[1:2] = [1.0; 1.0], [tunable = false, uncertain=true]
    p2[1:2] = [0.4; 0.2], [tunable = false]
end

@named lotka_volterra = System(
    [
        D(x(t)) ~ x(t) - p1[1] * x(t) * y(t) - p2[1] * x(t) * u(t),
        D(y(t)) ~ -y(t) + p1[2] * x(t) * y(t) - p2[2] * y(t) * u(t)
    ], t, [x(t), y(t), u(t)], [p1, p2];
    costs=Num[∫((x(t) - 1)^2 + (y(t) - 1)^2)],
    #costs= ∫.([(x(t) - 1)^2  (y(t) - 1)^2; u(t) h1(t)]),
    observed = [h1(t) ~ x(t); h2(t) ~ y(t)],
    constraints = [∫(h1(t)) ≲ 4.0; ∫(h2(t)) ≲ 4.0],
    consolidate=(x...)->first(x)[1], # Hacky, IDK what this is at the moment
)

costs = ∫.([(x(t) - 1)^2  (y(t) - 1)^2; u(t) h1(t)])


N = 24
shooting_points = [0., 6.0]
grid = ShootingGrid(shooting_points)

tpoints = collect(LinRange(0.,  12.0, N+1))[1:end-1]
controlmethod = DirectControlCallback(
    u(t) => (; timepoints=tpoints,
        defaults= collect(LinRange(0., 1., N))),
    h1(t) => (; timepoints=tpoints,
        defaults= ones(N), bounds=(0.,1.)),
    h2(t) => (; timepoints=tpoints,
        defaults= ones(N)
    )
)

builder = OEDProblemBuilder(
    lotka_volterra, controlmethod, grid, ACriterion(tspan),
         ForwardSolveInitialization()
)

# Instantiates the problem fully
builder = builder()

optfun = OptimizationProblem{true}(builder, AutoForwardDiff(), Tsit5())

# Initial plot
callback(x,l) = begin
    sol = optfun.f.f.predictor(x.u, saveat = 0.01)[1];
    f = Figure()
    plot!(Axis(f[1,1]), sol, idxs = [:x, :y, :u])
    plot!(Axis(f[1,2]), sol, idxs = [:G11, :G12, :G21, :G22])
    plot!(Axis(f[2,1]), sol, idxs = [:F11, :F12, :F22])
    plot!(Axis(f[2,2], xticks=0:1:12, limits=(nothing, (-0.02,1.02))), sol, idxs = [:w1, :w2])
    display(f)
    return false
end

callback((;u=optfun.u0),nothing)

# Optimize
sol = solve(optfun, Ipopt.Optimizer(); max_iter = 75, callback=callback,
         tol = 1e-6, hessian_approximation="limited-memory", )

sol = solve(optfun, BlockSQPOpt(); maxiters = 25, opttol = 1e-6, #callback=callback,
        options=blockSQP.sparse_options(), sparsity=optfun.f.f.predictor.permutation.blocks)

prob_before_fail = remake(optfun, u0=u_fail)
print_cb = (x,l) -> begin
    println(x.u)
    return false
end
sol = solve(prob_before_fail, BlockSQPOpt(); maxiters = 2, opttol = 1e-6, callback=print_cb,
        options=blockSQP.sparse_options(), sparsity=optfun.f.f.predictor.permutation.blocks)

u_fail = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.7, 0.0, 0.0, 1.0, 0.0, 0.0, 0.03617980992290208, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.26581499943553316, 0.016233338536982226, 0.0, 1.0, 0.0, 0.5265010359542992, 1.0, 0.0, 1.0, 0.8144558743629069, 0.0, 1.3297277644195908, -3.6070184162088528, -3.5054329813775476, 0.15864559950380475, 8.329165033466492, -1.2257704540850045, 13.840760489553945, 1.5422854018779415, 1.9598504744621337, 0.9984452852308942, 1.0, 0.0, 0.8817672342175056, 0.44682686538751865, 0.0940522808614258, 0.5198992611995329, 0.10168908844782176, 0.4107974118208349, 0.21994002029536014, 0.175288342624119, 0.5781484960127107, 0.06724334844568558, 0.1791437167186171, 0.6702401283109864, 0.010110853049250036, 0.0, 0.7198629695321384, 0.0, 0.0, 0.7332258767724932, 0.0, 0.0, 0.7035708016005642, -0.03232669505983303, -0.03760012440577783, 0.12556116319187954, 0.036236236027494435, 16.23239565320639, -0.7119567930354331, 19.505918080798462, 0.5411983235270208, 0.1906065128196286, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7760929616074795, 0.016781113023489558, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.3898833240069197, -3.743123067022206, -1.6913463620123652, -3.4852000450464127, 32.926615633538255, -6.58179372630226, 88.69749248443769, 0.5067784443741392, 3.3865596767544535, 0.0]

using DifferentiationInterface
jac_init = DifferentiationInterface.jacobian(x-> optfun.f.cons(x, nothing), AutoForwardDiff(), optfun.u0);
jac_fail = DifferentiationInterface.jacobian(x-> optfun.f.cons(x, nothing), AutoForwardDiff(), u_fail);

spy(jac_init)
spy(jac_fail)
spy(iszero.(jac_init) .- iszero.(jac_fail))

callback((;u=optfun.u0),nothing)
callback((;u=u_fail),nothing)

callback(sol, nothing)