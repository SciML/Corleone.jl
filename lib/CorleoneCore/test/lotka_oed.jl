using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using OrdinaryDiffEqTsit5
using CorleoneCore
using DifferentiationInterface
using Zygote
using CairoMakie
using SciMLSensitivity
using blockSQP
using Optimization
using OptimizationMOI, Ipopt
using LinearAlgebra

@variables x(..) = 0.5 y(..) = 0.7
@variables G11(..)=0.0 G12(..)=0.0 G21(..)=0.0 G22(..)=0.0
@variables z1(..)=0.0 z2(..)=0.0 F11(..)=0.0 F12(..)=0.0 F22(..)=0.0
@variables u(..)=0.3 [input = true] w1(..)=1.0 [input = true] w2(..)=1.0 [input = true]
@parameters p[1:4] = [1.0; 0.4; 1.0; 0.2]  [tunable=false] #y₀ = 2.0

_G = [G11(t) G12(t); G21(t) G22(t)]
_F = [F11(t); F12(t); F22(t)]
_z = [z1(t); z2(t)]

deqs = Num[x(t) - p[1] * x(t)*y(t) -  p[2] * x(t) * u(t);
                -y(t) + p[3] * x(t) * y(t) - p[4] * y(t) * u(t)]

senseqs = Symbolics.jacobian(deqs, [x(t),y(t)]) * _G .+ Symbolics.jacobian(deqs, [p[2]; p[4]])

Feqs = w1(t) * ([1 0] * _G)' * ([1 0] * _G) + w2(t) * ([0 1] * _G)' * ([0 1] * _G)

collect(senseqs)

ACrit = tr(inv(_F[1]*_F[3] - _F[2]^2) * [_F[3] -_F[2]; -_F[2] _F[1]])



tspan = (0.0,12.0)
@named lotka_oed = ODESystem(
    [
        D(x(t)) ~ deqs[1],
        D(y(t)) ~ deqs[2],
        D(_G[1,1]) ~ senseqs[1,1],
        D(_G[1,2]) ~ senseqs[1,2],
        D(_G[2,1]) ~ senseqs[2,1],
        D(_G[2,2]) ~ senseqs[2,2],
        D(z1(t)) ~ w1(t),
        D(z2(t)) ~ w2(t),
        D(F11(t)) ~ Feqs[1,1],
        D(F12(t)) ~ Feqs[1,2],
        D(F22(t)) ~ Feqs[2,2],
    ], t, [x(t), y(t), G11(t),  G12(t), G21(t), G22(t),
            z1(t),z2(t), F11(t), F12(t),F22(t), w1(t),w2(t), u(t)], [p];
    costs=Num[∀(ACrit, 12.0)],
    #consolidate=sum,
    tspan=tspan,
    #initialization_eqs=[
    #    y(0) ~ y₀
    #]
)

cstr = [z1(t)-4.0;
        z2(t)-4.0]

grid = ShootingGrid([6.0])
ns = length(grid.timepoints)+1

N = 12
nx = 3

controlmethod = IfElseControl(
    u(t)  => (; timepoints=LinRange(0.0,12.0,N+1)[1:end-1], defaults=.3 * ones(N)),
    w1(t) => (; timepoints=LinRange(0.0,12.0,N+1)[1:end-1], defaults=1.0 * ones(N)),
    w2(t) => (; timepoints=LinRange(0.0,12.0,N+1)[1:end-1], defaults=1.0 * ones(N)),
)

ctrl_sys = complete(grid(tearing(controlmethod(lotka_oed)); initializer=[F11(t) => 4.0 *ones(ns),
                                                                        F22(t) => 4.0 * ones(ns),
                                                                        F12(t) => 0.1 * ones(ns),
                                                                        z1(t) => 2.0 * ones(ns),
                                                                        z2(t) => 2.0 * ones(ns),
                                                                        ]))


pred = OCPredictor{true}(ctrl_sys, Tsit5(); adaptive=false, dt=0.1);

p0 = CorleoneCore.get_p0(pred)
p0_perm = CorleoneCore.get_p0(pred; permute=true)

sol = pred(p0; permute=false)
sol_perm = pred(p0_perm, permute=true)

obs = CorleoneCore.ObservedFunctions(pred.problem, ((expression=x(t), saveats=sol.time),
                                                    (expression=cstr, saveats=[12.0]),
                                                    (expression =ACrit, saveats=[12.0],))...)

obs(sol)

tu = controlmethod.controls[1].timepoints |> collect
tu = vcat(tu, 12.0)

function plotsol(sol)
    f = Figure()
    ax = Axis(f[1,1], xticks=0:3:12)
    ax2 = Axis(f[2,1], xticks=0:3:12)
    #_u = sol.states[p0_perm .== 0.3]
    #_u = vcat(_u, _u[end])
    ax3 = Axis(f[3,1], xticks=0:3:12, limits=(0,12,0,4.1))
    ax4 = Axis(f[4,1], xticks=0:3:12)
    [scatterlines!(ax, sol.time, x) for x in eachrow(sol.states[1:2,:])]
    [scatterlines!(ax2, sol.time, x) for x in eachrow(sol.states[3:6,:])]
    [scatterlines!(ax3, sol.time, x) for x in eachrow(sol.states[7:8,:])]
    [scatterlines!(ax4, sol.time, x) for x in eachrow(sol.states[9:11,:])]
    f
end

plotsol(sol)
plotsol(sol_perm)


gen_obj(permute::Bool) = begin
    obj = let pred=pred
        (p,x) -> begin
            sol = pred(p; permute=permute)
            obs(sol)[end]
        end
    end
    return obj
end

gen_cons(permute::Bool) = begin
    cons = let pred=pred
        (p,x_) -> begin
            sol= pred(p; permute=permute)
            measure_constraints = obs(sol)[end-1]
            shooting_constraints = reduce(vcat, [(-).(x...) for x in sol.shooting_variables])
            vcat(shooting_constraints, measure_constraints)
        end
    end
    return cons
end

obj = gen_obj(false)
obj_perm = gen_obj(true)

obj(p0, [])
obj_perm(p0_perm, [])

cons = gen_cons(false)
cons_perm = gen_cons(true)

cons(p0, [])

lb_perm = zeros(length(p0_perm))
lb_perm[p0_perm .== 0.5 .|| p0_perm .== 0.7] .== 0.1 # shooting xy
lb_perm[p0_perm .== 0.0] .== -50.0  # shooting G, F

ub_perm = ones(length(p0_perm))
ub_perm[p0_perm .== 0.0] .== 50.0
ub_perm[p0_perm .== 0.5 .|| p0_perm .== 0.7] .== 5.0
ub_perm[p0_perm .== 4.0 .|| p0_perm .== 0.1] .== 50.0
ub_perm[p0_perm .== 2.0] .== 4.0



num_cons = length(cons_perm(p0_perm, []))
optfun_perm = OptimizationFunction(obj_perm, Optimization.AutoFiniteDiff(), cons=(res,x,p) -> res .= cons_perm(x,p))
optprob_perm = OptimizationProblem(optfun_perm, p0_perm, Float64[], lcons = zeros(num_cons),
                ucons = vcat(zeros(num_cons-2), 4 * ones(2)) , lb=lb_perm, ub=ub_perm)
optsol_perm_b = solve(optprob_perm, BlockSQPOpt(); options=blockSQP.sparse_options(),
                    nlinfeastol=1e-6, opttol=1e-6,
                    sparsity=pred.permutation.blocks,  maxiters = 100)

optsol_perm_b = solve(optprob_perm, Ipopt.Optimizer();
                    tol=1e-6, hessian_approximation="limited-memory",
                      max_iter = 50)

sol = pred(optsol_perm_b.u)
plotsol(sol)