using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using OrdinaryDiffEq
using CorleoneCore
using DifferentiationInterface
using Zygote
using CairoMakie
using SciMLSensitivity
using blockSQP
using Optimization
using OptimizationMOI, Ipopt
using LinearAlgebra

@variables x(..) = 0.5 [bounds=(0.3,10.0)] y(..) = 0.7 [bounds=(0.3,10.0)]
@variables G11(..)=0.0 G12(..)=0.0 G21(..)=0.0 G22(..)=0.0
@variables z1(..)=0.0 [bounds=(0.0,12.0)] z2(..)=0.0 [bounds=(0.0,12.0)]
@variables F11(..)=1.0 F12(..)=0.0 F22(..)=1.0
@variables u(..)=0.3 [input = true, bounds=(0,1)] w1(..)=1.0 [input = true, bounds=(0,1)]
@variables w2(..)=1.0 [input = true, bounds=(0,1)]
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
        D(F11(t)) ~ Feqs[1,1],
        D(F12(t)) ~ Feqs[1,2],
        D(F22(t)) ~ Feqs[2,2],
        D(z1(t)) ~ w1(t),
        D(z2(t)) ~ w2(t),
        #D(w1(t)) ~ 0,
        #D(w2(t)) ~ 0,
        #D(u(t)) ~ 0,
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
prob = ODEProblem(complete(structural_simplify(lotka_oed)); allow_cost=true)

grid = ShootingGrid([6.0])
forward_sol = Array(solve(prob, RK4(); adaptive=false, dt=0.2, saveat=vcat(grid.timepoints,last(tspan))))


ns = length(grid.timepoints)+1

N = 24
nx = 3

grid_control = collect(0:12/N:12.0)[1:end-1]
controlmethod = IfElseControl(
    u(t)  => (; timepoints=grid_control, defaults=0.3 * ones(N)),
    w1(t) => (; timepoints=grid_control, defaults=0.3 * ones(N)),
    w2(t) => (; timepoints=grid_control, defaults=0.3 * ones(N)),
)

ctrl_sys = complete(grid(tearing(controlmethod(lotka_oed)) )) ###; initializer=[
                                                                #        x(t) => forward_sol[1,:],
                                                                #        y(t) => forward_sol[2,:],
                                                                #        G11(t) => forward_sol[3,:],
                                                                #        G12(t) => forward_sol[4,:],
                                                                #        G21(t) => forward_sol[5,:],
                                                                #        G22(t) => forward_sol[6,:],
                                                                #        F11(t) => forward_sol[7,:],
                                                                #        F22(t) =>  forward_sol[9,:],
                                                                #        F12(t) =>  forward_sol[8,:],
                                                                #        z1(t) => [2.0; 4.0],
                                                                #        z2(t) => [2.0; 4.0],
                                                                #        ]))


pred = OCPredictor{false}(ctrl_sys, Tsit5(); tstops=grid_control);

pred.permutation

p0 = CorleoneCore.get_p0(pred)
p0_perm = CorleoneCore.get_p0(pred; permute=true)

lb, ub = CorleoneCore.get_bounds(pred)
lb_perm, ub_perm = CorleoneCore.get_bounds(pred; permute=true)

sol = pred(p0; permute=false)
sol_perm = pred(p0_perm, permute=true)

function plotsol(sol)

    obs_u = CorleoneCore.ObservedFunctions(pred.problem, ((expression=u(t), saveats=sol.time),
                                                        (expression=w1(t), saveats=sol.time),
                                                        (expression=w2(t), saveats=sol.time))...;
                                                        aggregation=Base.Fix1(reduce, hcat))
    opt_comtrols = obs_u(sol)
    f = Figure()
    ax = Axis(f[1,1], xticks=0:3:12, title="States")
    ax2 = Axis(f[2,1], xticks=0:3:12, title="Sensitivities")
    ax3 = Axis(f[3,1], xticks=0:3:12, title="FIM")
    ax4 = Axis(f[1,2], xticks=0:3:12, title="u(t)")
    ax5 = Axis(f[2,2], xticks=0:3:12, title="w(t)")
    [scatterlines!(ax, sol.time, x) for x in eachrow(sol.states[1:2,:])]
    [scatterlines!(ax2, sol.time, x) for x in eachrow(sol.states[3:6,:])]
    [scatterlines!(ax3, sol.time, x) for x in eachrow(sol.states[7:9,:])]
    [scatterlines!(ax4, sol.time, x) for x in eachcol(opt_comtrols[:,1:1])]
    [scatterlines!(ax5, sol.time, x) for x in eachcol(opt_comtrols[:,2:3])]
    f
end

plotsol(sol)
plotsol(sol_perm)

obs = CorleoneCore.ObservedFunctions(pred.problem, ((expression=cstr, saveats=[12.0]),
                                                    (expression =ACrit, saveats=[12.0],))...)

calculate_cons(sol) = obs.observed[1].observed(sol.states[:,end], sol.parameters, sol.time[end])
calculate_Acrit(sol) = obs.observed[2].observed(sol.states[:,end], sol.parameters, sol.time[end])

calculate_Acrit(sol)
calculate_cons(sol)

gen_obj(permute::Bool) = begin
    obj = let pred=pred
        (p,x) -> begin
            sol = pred(p; permute=permute)
            calculate_Acrit(sol)
        end
    end
    return obj
end

gen_cons(permute::Bool) = begin
    cons = let pred=pred
        (p,x_) -> begin
            sol= pred(p; permute=permute)
            measure_constraints = calculate_cons(sol)
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

c0 = cons(p0, [])
num_cons = length(c0)

lb_cons = zeros(length(c0))
lb[end-1:end] .= -Inf

ub_cons = zeros(length(c0))

optfun = OptimizationFunction(obj, Optimization.AutoZygote(), cons=(res,x,p) -> res .= cons(x,p))
optprob = OptimizationProblem(optfun, p0_perm, Float64[], lcons = lb_cons,
                ucons = ub_cons , lb=lb, ub=ub)


optsol_ipopt = solve(optprob, Ipopt.Optimizer();
                    tol=1e-6, hessian_approximation="limited-memory",
                      max_iter = 150)

optsol_bspq = solve(optprob, BlockSQPOpt();
                    opttol=1e-6, maxiters = 50)

optfun_perm = OptimizationFunction(obj_perm, Optimization.AutoZygote(), cons=(res,x,p) -> res .= cons_perm(x,p))
optprob_perm = OptimizationProblem(optfun_perm, p0_perm, Float64[], lcons = lb_cons,
                ucons = ub_cons , lb=lb_perm, ub=ub_perm)



optsol_perm_b = solve(optprob_perm, BlockSQPOpt(); options=blockSQP.sparse_options(),
                    nlinfeastol=1e-6, opttol=1e-6,
                    sparsity=pred.permutation.blocks,  maxiters = 100)

optsol_perm_b = solve(optprob_perm, Ipopt.Optimizer();
                    tol=1e-6, hessian_approximation="limited-memory",
                      max_iter = 150)

sol = pred(optsol_perm_b.u; permute=true)
plotsol(sol)