using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using CorleoneCore
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using OrdinaryDiffEq
using Test

# Lotka
@variables begin
    x(..) = 0.5, [tunable=false]
    y(..) = 0.7, [tunable = false]
    u(..)=0.0, [input = true, bounds = (0., 1.)]
    h1(..)=0.0, [tunable=false]
    h2(..)=0.0, [tunable=false]
end
@parameters begin
    p1[1:2] = [1.0; 1.0], [tunable = false]
    p2[1:2] = [0.4; 0.2], [uncertain=true]
end
∫ = Symbolics.Integral(t in (0., 12.))

@named lotka_volterra = System(
    [
        D(x(t)) ~ x(t) - p1[1] * x(t) * y(t) - p2[1] * x(t) * u(t),
        D(y(t)) ~ -y(t) + p1[2] * x(t) * y(t) - p2[2] * y(t) * u(t)
    ], t, [x(t), y(t), u(t)], [p1, p2];
    costs=Num[∫((x(t) - 1)^2 + (y(t) - 1)^2)],
    observed = [h1(t) ~ x(t); h2(t) ~ y(t)],
    consolidate=(x...)->first(x)[1], # Hacky, IDK what this is at the moment
)
N = 24
shooting_points = [0., 6., 12.0]
grid = ShootingGrid(shooting_points, DefaultsInitialization())
controlmethod = DirectControlCallback(
    u(t) => (; timepoints=collect(LinRange(0.,  12.0, N)),
        defaults= collect(LinRange(0., 1., N))),
    h1(t) => (; timepoints=collect(LinRange(0.,  12.0, N)),
        defaults= ones(N)),
    h2(t) => (; timepoints=collect(LinRange(0.,  12.0, N)),
        defaults= ones(N)
    )
)


#builder = CorleoneCore.OCProblemBuilder(
#    lotka_volterra, controlmethod, grid, DefaultsInitialization()
#)
#builder= builder()


builder = CorleoneCore.OEDProblemBuilder(
    lotka_volterra, controlmethod, grid, DCriterion(), DefaultsInitialization(), Dict()
)

# Instantiates the problem fully
builder = builder()


using SciMLSensitivity, Optimization, OptimizationMOI, Ipopt
using SciMLSensitivity.ForwardDiff, SciMLSensitivity.Zygote, SciMLSensitivity.ReverseDiff
using CairoMakie
optfun = OptimizationProblem{true}(builder, AutoForwardDiff(), Tsit5())

# Initial plot
blocks = optfun.f.f.predictor.permutation.blocks
sol = optfun.f.f.predictor(optfun.u0, saveat = 0.01)[1];
f = Figure()
plot!(Axis(f[1,1]), sol, idxs = [:x, :y, :u])
plot!(Axis(f[1,2]), sol, idxs = [:G11, :G12, :G21, :G22])
plot!(Axis(f[2,1]), sol, idxs = [:F11, :F12, :F22])
plot!(Axis(f[2,2]), sol, idxs = [:w1, :w2])
f

# Optimize
sol = solve(optfun, Ipopt.Optimizer(); max_iter = 150,
         tol = 1e-6, hessian_approximation="limited-memory", )

# Result plot
pred = optfun.f.f.predictor(sol.u, saveat = 0.01)[1]
f = plot(pred, idxs = [:x, :y, :u])


fil = findall(CorleoneCore.is_statevar, unknowns(builder.system))
sts = unknowns(builder.system)[fil]
ff = equations(builder.system)[fil]

_ff = map(ff) do eq
    if operation(eq.lhs) == Differential(t)
        eq.rhs
    else
        eq.lhs - eq.rhs
    end
end

dfdx = Symbolics.jacobian(_ff, sts)

parameters(builder.system)
pp = filter(x -> istunable(x) && !(CorleoneCore.is_shootingvariable(x) || CorleoneCore.is_shootingpoint(x) || CorleoneCore.is_localcontrol(x) || CorleoneCore.is_tstop(x)), parameters(builder.system))
pp = reduce(vcat, pp)
np = length(pp)
dfdp = Symbolics.jacobian(_ff, pp)

@variables G(..)[1:2,1:np] = 0.0
_G = collect(G(t))
dg = vec(dfdp .+ dfdx * _G)


sens_eqs = vec(_G) .~

@variables F(..)[1:np,1:np] = 0.0
_F = collect(F(t))

h = [x(t); y(t)]

@variables w(..)[1:2] = 0.0 [input=true, bounds=(0,1)]

F_eq = map(enumerate(h)) do (i,h_i)
    hix = Symbolics.jacobian([h_i], sts)
    @info hix
    gram = hix * _G
    w(t)[i] * gram' * gram
end |> sum

fisher_eqs = vec(_F) .~ vec(F_eq)

new_eqs = reduce(vcat, [sens_eqs, fisher_eqs])

oedsys = System(
    new_eqs,
    ModelingToolkit.get_iv(lotka_volterra),
    [G(t), F(t), w(t)], [],
    name = nameof(lotka_volterra),
    )


newsys = CorleoneCore.extend_system(lotka_volterra, oedsys)

prob = ODEProblem(complete(mtkcompile(newsys)), [], (0.0,12.0); allow_cost=true, build_initializeprob=false)
solve()