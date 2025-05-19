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

@variables x(..) = 0.5 y(..) = 0.7 u(..) [input = true]
@parameters p[1:4] = [1.0; 0.4; 1.0; 0.2]  [tunable=false] #y₀ = 2.0
tspan = (0.0,12.0)
@named first_order = ODESystem(
    [
        D(x(t)) ~ x(t) - p[1] * x(t)*y(t) -  p[2] * x(t) * u(t),
        D(y(t)) ~ -y(t) + p[3] * x(t) * y(t) - p[4] * y(t) * u(t)
    ], t, [x(t), y(t), u(t)], [p];
    costs=Num[∫((x(t) - 1)^2 + (y(t)-1)^2)],
    consolidate=sum,
    tspan=tspan,
    #initialization_eqs=[
    #    y(0) ~ y₀
    #]
)



grid = ShootingGrid([4.0, 8.0])
ns = sum(grid.timepoints .> 0.0 .&&  grid.timepoints .< 12.0) + 1

N = 36
nx = 3
controlmethod = IfElseControl(
    u(t) => (; timepoints=LinRange(0,12,N+1)[1:end-1], defaults=.3 * ones(N))
)

#controlmethod = DirectControlCallback(
#    u(t) => (; timepoints=LinRange(0,12,N+1)[1:end-1], defaults=.3 * ones(N))
#)

newsys = CorleoneCore.extend_costs(first_order)
ctrl_sys = complete(grid(tearing(controlmethod(newsys))))#; initializer=[x(t) => rand(ns), y(t) => rand(ns)]))

ukws = ModelingToolkit.unknowns(ctrl_sys)
CorleoneCore.is_costvariable.(ukws)

pred = OCPredictor{false}(ctrl_sys, Tsit5(); abstol=1e-10, reltol=1e-10);

p0 = CorleoneCore.get_p0(pred)
p0_perm = CorleoneCore.get_p0(pred; permute=true)

sol = pred(p0; permute=false)
sol_perm = pred(p0_perm, permute=true)


function plotsol(sol)
    f = Figure()
    ax = Axis(f[1,1], xticks=0:3:12, limits=(0,12,0,4))
    [scatterlines!(ax, sol.time, x) for x in eachrow(sol.states)]
    f
end

plotsol(sol)
plotsol(sol_perm)

gen_obj(permute::Bool) = begin
    obj = let pred=pred
        (p,x) -> begin
            sol = pred(p; permute=permute)
            sum(sol.mayer_variables)[1]
        end
    end
    return obj
end

gen_cons(permute::Bool) = begin
    cons = let pred=pred
        (p,x_) -> begin
            sol= pred(p; permute=permute)
            reduce(vcat, [(-).(x...) for x in sol.shooting_variables])
        end
    end
    return cons
end

obj = gen_obj(false)
obj_perm = gen_obj(true)

obj_perm(p0_perm, [])

cons = gen_cons(false)
cons_perm = gen_cons(true)

grad = DifferentiationInterface.gradient(y -> obj_perm(y,[]), AutoZygote(), p0_perm)

obj(p0,[]) == obj_perm(p0_perm, [])
cons(p0,[]) == cons_perm(p0_perm,[])

hess = DifferentiationInterface.hessian(y -> obj_perm(y,[]) + sum(cons_perm(y,[])), AutoZygote(), p0_perm)
spy(hess)
hess_sym = Symmetric(hess)
spy(hess_sym)

lb = zeros(size(p0))
lb[p0 .== 0.5 .|| p0 .== 0.7] .= 0.5
ub = 5 * ones(size(p0))
ub[p0 .== 0.3] .= 1.0

lb_perm = lb[pred.permutation.fwd]
ub_perm = ub[pred.permutation.fwd]

num_cons = length(cons(p0,[]))

function callback(x,l; doplot=false, permutation=collect(1:length(x.u)))
    u_perm = x.u[permutation]
    _sol = pred(u_perm)
    f= Figure()
    ax1 = Axis(f[1,1], limits=(nothing, (0,3)), title="States", xticks=0:3:12)
    ax2 = Axis(f[2,1], title="Controls", xticks=0:3:12)
    ax3 = Axis(f[3,1], yscale=log10, limits=(nothing, nothing,1e-12, 1e1), yticks=([10^i for i=-12.0:4.0:1.0], ["1e-12", "1e-8", "1e-4", "1e-2"]), title="|Matching conditions|", xticks=0:3:12)
    linkxaxes!(ax1, ax2, ax3)
    [scatterlines!(ax1, _sol.time, x) for x in eachrow(_sol.states)]
    matching_ = abs.(reduce(hcat, [(-).(x...) for x in _sol.shooting_variables]))
    t_shooting = unique(last.(pred.shooting_intervals))
    [scatterlines!(ax3, t_shooting, x) for x in eachrow(matching_)]

    t_u = vcat(controlmethod.controls[1].timepoints, last(first_order.tspan))
    _u = u_perm[p0 .== 0.3]
    stairs!(ax2, t_u, vcat(_u, _u[end]), step=:post)
    display(f)
    if doplot
        counter = x.iter
        plotdir = joinpath(@__DIR__, "plots")
        isdir(plotdir) || mkdir(plotdir)
        CairoMakie.save(joinpath(plotdir, "fig_$counter.png"), f)
    end
    return false
end

p1 = copy(p0)
p1[p0 .== 0.3] .= 0.5

optfun = OptimizationFunction(obj, Optimization.AutoZygote(), cons=(res,x,p) -> res .= cons(x,p))
optprob = OptimizationProblem(optfun, p0, Float64[], lcons = zeros(num_cons),
                ucons = zeros(num_cons), lb=lb, ub=ub)
optsol_i = solve(optprob, Ipopt.Optimizer(); hessian_approximation="limited-memory",
            max_iter = 50, callback=callback)
optsol_b = solve(optprob, BlockSQPOpt();  globalization=1, nlinfeastol=1e-6, maxiters = 100)
callback(optsol_b, optsol_b.objective; doplot=false)

optfun_perm = OptimizationFunction(obj_perm, Optimization.AutoZygote(), cons=(res,x,p) -> res .= cons_perm(x,p))
optprob_perm = OptimizationProblem(optfun_perm, p0_perm, Float64[], lcons = zeros(num_cons) .- 1e-6,
                ucons = zeros(num_cons) .+ 1e-6 , lb=lb_perm, ub=ub_perm)
optsol_perm_b = solve(optprob_perm, BlockSQPOpt(); options=blockSQP.sparse_options(), nlinfeastol=1e-6,
                    sparsity=pred.permutation.blocks, maxiters = 100)

callback(optsol_perm_b, optsol_perm_b.objective; permutation=pred.permutation.rev, doplot=false)

# Christoph has the code for this in Mattermost
# Was done via Vibe Coding, so there will be errors here (probably)
perm = cuthill_mckee(adj)
spy(adj[perm, reverse(perm)])


using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using OrdinaryDiffEqTsit5, SciMLSensitivity, SciMLStructures
using DifferentiationInterface
using Zygote, Zygote.ForwardDiff

@variables x(t) = 1.0
@parameters p = 1.0
@named sys = ODESystem([D(x) ~ -p * x], t)
prob = ODEProblem{false}(structural_simplify(sys), [], (0.0, 1.0), [], build_initializeprob=false, jac=true)
p0, replace, alias = SciMLStructures.canonicalize(SciMLStructures.Tunable(), prob.p)

predict = let prob = prob, replace = replace
    (p) -> begin
        p = replace(p)
        prob = remake(prob, p=p)
        sol = solve(prob, Tsit5())
        sum(abs2, sol[end])
    end
end

# Works
predict(p0)

# Works
DifferentiationInterface.jacobian(predict, AutoZygote(), p0)

# Fails with various configs here
DifferentiationInterface.hessian(predict, AutoForwardDiff(), p0)

DifferentiationInterface.hessian(predict, SecondOrder(AutoForwardDiff(), AutoZygote()), p0)

DifferentiationInterface.hessian(predict, AutoZygote(), p0)


using DifferentiationInterface
using Zygote.ForwardDiff

DifferentiationInterface.hessian(testme, SecondOrder(AutoZygote(), AutoForwardDiff()), p0)

testme(p0)

using Zygote, SciMLSensitivity

Zygote.gradient(testme, p0)

struct OCPredictor{P,}
end


using Zygote



Zygote.gradient(p -> sum(u0([1.0, 2.0], p, 0.0)), ps)

Zygote.gradient(p -> sum(shooter([1.0, 2.0], p, 1.0)), ps)


f = ODEFunction(ctrl_sys)

f([1.0], ps, 1.1)

shooter([], ps, 2.0)

# ctrl_sys = CorleoneCore.extend_costs(first_order)
# ctrl_sys = controlmethod(ctrl_sys)
# ctrl_sys = CorleoneCore.change_of_variables(ctrl_sys)

unknowns(ctrl_sys)
observed(ctrl_sys)
parameters(ctrl_sys)
equations(ctrl_sys)
ModelingToolkit.get_costs(ctrl_sys)
discrete_events(ctrl_sys)
initialization_equations(ctrl_sys)

function build_initializer(sys)
    init_eqs = initialization_equations(sys)
    lhs = map(x -> x.lhs, init_eqs)
    rhs = map(x -> x.rhs, init_eqs)
    return first(ModelingToolkit.generate_custom_function(sys, rhs))
end

f_ = build_initializer(complete(ctrl_sys)) |> eval

f_([], prob.p, 2.0)


defaults(ctrl_sys)
unknowns(ctrl_sys)

prob = ODEProblem(complete(ctrl_sys), [], (0.0, 10.0), allow_cost=true, build_initializeprob=false)

using BenchmarkTools

@btime solve($prob, $(Tsit5()), tstops=$tstops);

sol = solve(prob, Tsit5(), tspan=(0.0, 2.0));

f = CairoMakie.Figure()
ax1 = CairoMakie.Axis(f[1, 1], title="States")
plot!(ax1, sol, idxs=[ctrl_sys.x])
ax2 = CairoMakie.Axis(f[2, 1], title="Controls")
plot!(ax2, sol, idxs=[ctrl_sys.u])
#ax3 = CairoMakie.Axis(f[3, 1], title = "Objective", xlabel = "t")
#plot!(ax3, sol, idxs = ModelingToolkit.get_costs(prob.f.sys))
linkxaxes!(ax1, ax2,)
display(f)

using SciMLStructures
using SciMLSensitivity, Zygote
using ChainRulesCore

p, remake, alias = SciMLStructures.canonicalize(SciMLStructures.Tunable(), prob.p)
p0 = rand(Float64, size(p))

pred = let prob = prob, remake = remake, tstops = tstops, f_ = f_
    (p) -> begin
        ps = remake(p)
        u0 = f_([], ps, 1.0)
        prob_ = SciMLBase.remake(prob, u0=u0, p=ps, tspan=(1.0, 2.0))
        sum(Array(solve(prob_, Tsit5(),)))
    end
end

pred(p0)

Zygote.gradient(pred, rand(14))



@parameters A[1:2, 1:3]

using LinearAlgebra

obj = det(∫.([x u 0 1; 0 x^2 u+x 1; 0 x 0 1; sin(x) 0 0 1]))

newsys, expr = CorleoneCore.extend_integrals(ctrl_sys, obj)

expr

@variables x(t)
using LinearAlgebra
b = [x^2 1; 1 0]
A = ∫.([x 0; x+2 sin(x)])
ex = det(∫.(b * A))


equations(newsys)
obj = ∀(x - data, data)

∀(sin(x) + p, [0.1, 0.2, 0.3, 1.0])

@variables x(t) [bounds = (0.0, 1.0)]
@parameters p = 1.0 [bounds = (-1.0, 2.0)] x₀ = 1.0 [bounds = (0.0, 1.0)]

# This is type unstable, but will hopefully be fixed soon enough
@named sys = ODESystem(
    [D(x) ~ p * x], t, [x], [p,],
)

function extend_initialization(sys)
    tunables = filter(!Base.Fix2(Symbolics.hasmetadata, Symbolics.VariableDefaultValue), unknowns(sys))
    ics = []
    ic_eqs = Equation[]
    foreach(tunables) do u
        varsym = Symbol(operation(u), Symbol(Char(0x2080)))
        bounds = ModelingToolkit.getbounds(u)
        def = (bounds[2] - bounds[1]) / 2
        u0 = first(@parameters ($varsym) = def [bounds = bounds])
        push!(ics, u0)
        push!(ic_eqs, u ~ u0)
    end
    init_sys = ODESystem(
        Equation[], t, [], ics, initialization_eqs=ic_eqs,
        name=Symbol(nameof(sys), :_, :initialization)
    )
    extend(sys, init_sys)
end

# This is for shooting and NOT READY YET
function extend_initialization(sys, timepoints::AbstractVector)
    tspan = ModelingToolkit.get_tspan(sys)

    tunables = filter(!Base.Fix2(Symbolics.hasmetadata, Symbolics.VariableDefaultValue), unknowns(sys))
    ics = []
    ic_eqs = Equation[]
    foreach(tunables) do u
        varsym = Symbol(operation(u), :ₜ, Symbol(Char(0x2080)))
        bounds = ModelingToolkit.getbounds(u)
        def = (bounds[2] - bounds[1]) / 2
        u0 = first(@parameters ($varsym) = def [bounds = bounds])
        push!(ics, u0)
        push!(ic_eqs, u ~ u0)
    end
    init_sys = ODESystem(
        Equation[], t, [], ics, initialization_eqs=ic_eqs,
        name=Symbol(nameof(sys), :_, :initialization)
    )
    extend(sys, init_sys)
end

sys = extend_initialization(sys)

prob = ODEProblem(structural_simplify(sys), [], (0.0, 1.0), [], build_initializeprob=true)
prob.u0

sol = solve(prob, Tsit5(), saveat=[0.0, 1.0])
plot(sol)

cons = CorleoneCore.extend_forall(
    sys,
    ∀(2x - 1 ≲ 1, [0.0, 1.0])...,
    p ≲ ∀(x^2, 1.0)
)

ts = Trajectory(sol)

cons(ts)

using SciMLStructures

p, remake, _ = SciMLStructures.canonicalize(SciMLStructures.Tunable(), prob.p)
p0 = rand(Float64, size(p))

@code_warntype solve(prob, Tsit5(), p=remake(p0))


p0 = rand(Float64, size(p))
sol2 = solve(prob, Tsit5(), p=remake(p0))
plot(sol2)

using SciMLSensitivity, Zygote

pred = let prob = prob, remake = remake
    (p) -> sum(Array(solve(prob, Tsit5(), p=remake(p))))
end

Zygote.gradient(pred, randn(2))
