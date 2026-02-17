using Corleone
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using OrdinaryDiffEqTsit5

@variables x(..) = 0.5 [tunable = false,] y(..) = 0.7 [tunable=false,]
@variables u(..) = 0.0 [bounds = (0.0, 1.0), input = true]
@constants begin
    c₁ = 0.4
    c₂ = 0.2
end
@parameters begin 
    α = 1.0, [tunable=false,]
    β = 1.0, [tunable=true, bounds = (0.9, 1.1)]
end

cost = [
    Symbolics.Integral(t in (0.0, 12.0))(
        (x(t) - 1.0)^2 + (y(t) - 1.0)^2 
    ),
]

cons = [
    x(0.0) ≳ 0.2,
    β ~ 1.0
]

@named lotka = System(
    [
        D(x(t)) ~ α*x(t) - β * x(t) * y(t) - c₁ * u(t) * x(t),
        D(y(t)) ~ - y(t) + x(t) * y(t) - c₂ * u(t) * y(t),
    ], t; costs = cost, constraints = cons
)

dynopt = CorleoneDynamicOptProblem(
    lotka, [],
    u(t) => 0.0:0.1:11.9,
    algorithm = Tsit5(),
    #shooting = [0.0, 5.0]
)

using ComponentArrays, ForwardDiff
using Optimization 
using OptimizationMOI, Ipopt
using LuxCore, Random

optprob = OptimizationProblem(dynopt, AutoForwardDiff(), Val(:ComponentArrays))

ps, st = LuxCore.setup(Random.default_rng(), dynopt.layer) 

traj, _ = dynopt.layer(nothing, ps, st)
vars = map(dynopt.getters) do get 
    get(traj)
end

dynopt.objective(ps, st)

optprob.f(optprob.u0, optprob.p)


sol = solve(
        optprob, Ipopt.Optimizer(), max_iter = 1000, tol = 5.0e-6,
        #hessian_approximation = "limited-memory"
    )


opttraj, _ = dynopt.layer(nothing, ComponentArray(sol.u ,optprob.f.f.ax), st)

using CairoMakie

plot(opttraj, idxs = [1, 2, 4])

using LuxCore, Random

ps, st = LuxCore.setup(Random.default_rng(), dynopt.layer)

dynopt.layer(nothing, ps, st)

using InfiniteOpt
using Ipopt

cas_optprob = InfiniteOptDynamicOptProblem(mtkcompile(lotka, inputs = [u(t)]), [x(t) => 0.5, y(t) => 0.7], (0.0, 12.0), dt = 0.01)
ca_sol = SciMLBase.solve(cas_optprob, InfiniteOptCollocation(Ipopt.Optimizer))

ps = @constants h_c m₀ h₀ g₀ D_c c Tₘ m_c
vars = ModelingToolkit.@variables h(t) v(t) m(t) = m₀ [bounds = (m_c, 1)] T(t)=1.0 [input = true, bounds = (0, 3.5)]
drag(h, v) = D_c * v^2 * exp(-h_c * (h - h₀) / h₀)
gravity(h) = g₀ * (h₀ / h)

eqs = [
    D(h) ~ v,
    D(v) ~ (T - drag(h, v)) / m - gravity(h),
    D(m) ~ -T / c,
]

(ts, te) = (0.0, 0.2)
costs = [-EvalAt(te)(h)]
cons = [EvalAt(te)(T) ~ 0, EvalAt(te)(m) ~ 0.6]
@named rocket = System(eqs, t, vars, ps; costs, constraints = cons)
u0map = [h => h₀, v => 0]
pmap = [
    g₀ => 1, m₀ => 1.0, h_c => 500, c => 0.5 * √(g₀ * h₀), D_c => 0.5 * 620 * m₀ / g₀,
    Tₘ => 3.5 * g₀ * m₀, T => 0.0, h₀ => 1, m_c => 0.6,
]

iprob = JuMPDynamicOptProblem(mtkcompile(rocket, inputs = [T]), [u0map; pmap], (ts, te); dt = 0.001)
isol = SciMLBase.solve(iprob, JuMPCollocation(Ipopt.Optimizer))

using CairoMakie 
f = Figure() 
ax1 = CairoMakie.Axis(f[1,1])
plot!(ax1, isol.sol, idxs = [h, v, m])
ax2 = CairoMakie.Axis(f[2,1])
stairs!(ax2, isol.input_sol)
display(f)


using OrdinaryDiffEq
cache = CorleoneDynamicOptProblem(rocket, [u0map; pmap], T => ts:0.001:te, algorithm = RadauIIA5())
ps, st = LuxCore.setup(Random.default_rng(), cache.layer)
traj, _ = cache.layer(nothing, ps, st)

plot(traj)
using Optimization, OptimizationMOI, ComponentArrays

p0, st = LuxCore.setup(Random.default_rng(), cache.layer)
p0 = ComponentArray(p0)

objective = let cache = cache, ax = getaxes(p0)
    (p, st) -> begin 
        p = ComponentArray(p, ax)
        traj, _ = cache.layer(nothing, p, st)
        vars = map(cache.getters) do _get 
            _get(traj)
        end
        cache.objective(vars...)
    end
end

cons = let cache = cache, ax = getaxes(p0)
    (res, p, st) -> begin 
        p = ComponentArray(p, ax)
        traj, _ = cache.layer(nothing, p, st)
        vars = map(cache.getters) do _get 
            _get(traj)
        end
        cache.contraints(res, vars...)
        return 
    end
end

@code_warntype objective(p0, st)
@code_warntype cons(zeros(2), p0, st)

using ForwardDiff
using Zygote 
using ReverseDiff 


optfun = OptimizationFunction{true}(
    objective, AutoForwardDiff(), 
    cons = cons, 
)

optprob = OptimizationProblem(optfun, collect(p0), st, ub = 3.5 .+ zeros(200), lb = zeros(200), ucons = cache.ucons, lcons = cache.lcons)

optsol = SciMLBase.solve(optprob, Ipopt.Optimizer(), 
    hessian_approximation = "limited-memory", 
    max_iter = 1000, )

optsol = SciMLBase.solve(remake(optprob, u0 = optsol.u), Ipopt.Optimizer(), 
    hessian_approximation = "limited-memory", 
    max_iter = 100, )

traj, _ = cache.layer(nothing, optsol.u .+ zero(p0), st)

f = plot(traj, idxs = [1,2,3])
stairs!(traj, idxs = [4])
display(f)

vars = map(cache.getters) do get
    get(traj)
end

@code_warntype cache.objective(vars...)

@code_warntype cache.contraints[1]([], vars...)


#prob = mtkcompile(new_sys, inputs = [u(t)])

prob = ODEProblem(prob, [], (0.0, 12.0), check_compatibility = false)

solve(prob, Tsit5())

prob = __preprocess_system(block, u(t) => 0.0:0.25:10.75, algorithm = Tsit5())


costs, constraints!, prob = preprocess_system(block, u(t) => 0.0:0.1:10.0, algorithm = Tsit5());
ps, st = LuxCore.setup(Random.default_rng(), prob)
traj, _ = prob(nothing, ps, st)

plot(traj, idxs = [1, 2, 3])


using OrdinaryDiffEqTsit5

using Corleone

layer = SingleShootingLayer(prob, Tsit5(), controls => ControlParameter(0.0:0.1))

sol = solve(prob, Tsit5())

using CairoMakie

plot(sol)


ODEProblem(mtkcompile(sys), [], (0.0, 10.0))

subs = Dict()

collect_integrals!(subs, cost[1], t)

ModelingToolkitBase.get_tspan(block)

objective = OptimalControlFunction{true}(
    costs, builder, alg, args...; consolidate = consolidate, kwargs...
)


sym = Symbolics.variable(:u; T = Array{Real, 2})


t = ModelingToolkit.t_nounits
D = ModelingToolkit.D_nounits

@parameters h_c m₀ h₀ g₀ D_c c Tₘ m_c
@variables begin
    h(..)
    v(..)
    m(..), [bounds = (m_c, 1)]
    T(..), [input = true, bounds = (0, Tₘ)]
end

drag(h, v) = D_c * v^2 * exp(-h_c * (h - h₀) / h₀)
gravity(h) = g₀ * (h₀ / h)

eqs = [
    D(h(t)) ~ v(t),
    D(v(t)) ~ (T(t) - drag(h(t), v(t))) / m(t) - gravity(h(t)),
    D(m(t)) ~ -T(t) / c,
]

(ts, te) = (0.0, 0.2)
costs = [-h(te)]
cons = [T(te) ~ 0, m(te) ~ m_c]

@named rocket = System(eqs, t; costs, constraints = cons)
rocket = mtkcompile(rocket, inputs = [T(t)])

u0map = [h(t) => h₀, m(t) => m₀, v(t) => 0]
pmap = [
    g₀ => 1, m₀ => 1.0, h_c => 500, c => 0.5 * √(g₀ * h₀), D_c => 0.5 * 620 * m₀ / g₀,
    Tₘ => 3.5 * g₀ * m₀, T(t) => 0.0, h₀ => 1, m_c => 0.6,
]
