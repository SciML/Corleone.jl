using Pkg
Pkg.activate(joinpath(pwd(), "lib/CorleoneCore"))

using CorleoneCore

using Pkg
Pkg.activate(pwd())

using OrdinaryDiffEqTsit5
using SciMLSensitivity
using LuxCore
using Random
using ComponentArrays

using CairoMakie
using BenchmarkTools
using Zygote
using SciMLSensitivity.ForwardDiff

function lotka_dynamics(du, u, p, t, c)
    du[1] = u[1] - prod(u) - 0.2 * c[1] * u[1]
    du[2] = -u[2] + prod(u) - 0.4 * c[2] * u[2]
    du[3] = (u[1]-1.0)^2 + (u[2] - 1.0)^2
end


tspans = ((0., 12.),)#(0., 6.), (nextfloat(6.), 12.))
Δt = 0.1

t0 = first(first(tspans))
tinf = last(last(tspans))

controls = SignalContainer(
    PiecewiseConstant(
        t0:Δt:(tinf-Δt)
    ),
    PiecewiseConstant(
        t0:Δt:(tinf-Δt)
    ),
    aggregation=Base.Fix1(reduce, vcat)
)

dyn = ControlledDynamics(lotka_dynamics, controls)

cdyn = CorleoneCore.wrap_model(Random.default_rng(), dyn)

u0 = [0.5, 0.7, 0.]
tspan = (t0, tinf)
p = ComponentArray(LuxCore.initialparameters(Random.default_rng(), dyn))
problem = ODEProblem(cdyn, u0, tspan, p)
layers = map(tspans) do tspani
    (gensym(), ShootingProblem(problem, tspan=tspani, tunable = Int[]))
end |> NamedTuple;

shootingproblems = CorleoneCore.MultipleShootingProblem(layers)
simlayer = CorleoneCore.SolverLayer(Tsit5(), EnsembleSerial(), shootingproblems)
ps, st = LuxCore.setup(Random.default_rng(), simlayer)

p = ComponentArray(ps)

sol, st = simlayer(problem, ps, st)
plot(sol)

using Optimization, OptimizationMOI, Ipopt

loss = let pax = ComponentArrays.getaxes(p), st = st, model = (p,st) -> simlayer(problem, p, st)
    (p, ::Any) -> begin 
        sol, _ =  model(ComponentArray(p, pax), st) 
        xs = map(Array, sol)
        last(xs)[end, end]#sum(sum(abs2, xi .- 1.0) for xi in xs)
    end
end

loss(collect(p), nothing)

grad_fd = ForwardDiff.gradient(Base.Fix2(loss, nothing), collect(p))

grad_zg = Zygote.gradient(Base.Fix2(loss, nothing), collect(p))[1]

optfun = OptimizationFunction(
    loss, AutoZygote(), 
)

optprob = OptimizationProblem(
    optfun, collect(p), lb = zero(collect(p)), ub = zero(collect(p)) .+ 1
)

optimizer = Ipopt.Optimizer(
)

uopt = solve(optprob, optimizer, 
     hessian_approximation = "limited-memory", max_iter = 1000
)


popt = ComponentArray(uopt) + zero(p)

sol, st = simlayer(problem, popt .+ zero(p), st)

plot(sol)


fff = scatter(grad_fd)

scatter!(grad_zg)
display(fff)