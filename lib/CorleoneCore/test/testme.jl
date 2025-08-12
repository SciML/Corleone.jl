using CorleoneCore
using Pkg
Pkg.activate(joinpath(pwd(), "../.."))

using OrdinaryDiffEqTsit5
using SciMLSensitivity
using LuxCore
using Random
using ComponentArrays

using CairoMakie

tstart = time()

controls = SignalContainer(
    PiecewiseConstant(
        LinRange(0., 10., 20)
    ), 
    PiecewiseConstant(
        LinRange(0., 5., 20)    
    ), 
    aggregation = Base.Fix1(reduce, vcat)
)


ps, st = LuxCore.setup(Random.default_rng(), controls)

ComponentVector(ps)


control_signal = CorleoneCore.wrap_model(controls, ps, st) 

control_signal((randn(2), 0.), ps)

function lotka_dynamics(u, p, t, c)
    [u[1] - prod(u) - 0.2 * c[1] * u[1],
     -u[2] + prod(u) - 0.4 * c[2] * u[2]]
end 

dyn = ControlledDynamics(lotka_dynamics, controls)
ps, st = LuxCore.setup(Random.default_rng(), dyn)

cdyn = CorleoneCore.wrap_model(dyn, ps, st)

ComponentVector(ps)

#lv = let control_signal = control_signal 
#function (du, u, p, t)
#    c = control_signal((u,t), p.controls,)
#    du[1] = u[1] - prod(u) - 0.2 * c[1] * u[1]
#    du[2] = -u[2] + prod(u) - 0.4 * c[2] * u[2]
#end
#end 


u0 = ones(2)
tspan = (0., 12.)
p = ComponentVector(ps)
du0 = zero(u0)

problem = ODEProblem(cdyn , u0, tspan, p)


shootingproblems = CorleoneCore.MultipleShootingProblem((;
    layer1 = ShootingProblem(remake(problem, tspan = (0., 6.))),
    layer2 = ShootingProblem(remake(problem, tspan = (6., 8.))),
    layer3 = ShootingProblem(remake(problem, tspan = (8., 12.)))
))

shootinglayer = CorleoneCore.SolverLayer(Tsit5(), EnsembleSerial(), shootingproblems)

ps, st = LuxCore.setup(Random.default_rng(), shootinglayer)

using BenchmarkTools 

@btime shootinglayer($problem, $ps, $st);

sol = shootinglayer(problem, ps, st)

f = CairoMakie.Figure()
ax = CairoMakie.Axis(f[1,1], title  = "x(t)")
X = Array(sol)

for i in axes(X, 1), j in axes(X, 3)
    scatterlines!(ax, sol[j].t, X[i, :, j])
end
display(f)


ax2 = CairoMakie.Axis(f[2,1], title = "c(t)")
u_t = reduce(hcat, map(x->Base.Fix2(control_signal, p.controls)((x,)), sol.t))
stairs!(ax2, sol.t, u_t[1, :])
stairs!(ax2, sol.t, u_t[2, :])
display(f)

using Zygote

loss = let prob = shootinglayer 
    function (p, st = nothing) 
        sol = solve(prob, Tsit5(), saveat = [12.0],  tspan = (5., 12.), p = p, sensealg = GaussAdjoint())
        sum(abs2, 1 .- Array(sol)[:, end])
    end
end

using BenchmarkTools

loss(p)

@btime loss($p);

using SciMLSensitivity.ForwardDiff 

H = ForwardDiff.hessian(loss, p)
spy(H)