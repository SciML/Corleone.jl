using Pkg
Pkg.activate(joinpath(@__DIR__, "../"))


using CorleoneCore
using OrdinaryDiffEq
using SciMLSensitivity
using ComponentArrays
using LuxCore
using Random

using CairoMakie
using BenchmarkTools
using Zygote
using ForwardDiff
using SciMLSensitivity.ReverseDiff

using Optimization
using OptimizationMOI
using Ipopt
using blockSQP

using SymbolicIndexingInterface

function lotka_dynamics!(du, u, p, t)
    du[1] = u[1] - p[2] * prod(u[1:2]) - 0.4 * p[1] * u[1];
    du[2] = -u[2] + p[3] * prod(u[1:2]) - 0.2 * p[1] * u[2];
    du[3] = (u[1]-1.0)^2 + (u[2] - 1.0)^2
end

tspan = (0., 12.)
u0 = [0.5, 0.7, 0.]
p0 = [0.0, 1.0, 1.0]
prob = ODEProblem(
    ODEFunction(lotka_dynamics!, 
    sys = SymbolCache(
        [:x, :y, :L], 
        [:w, :α, :β],
        [:t]
    ),
    ), u0, tspan, p0,
    #abstol = 1e-8, reltol = 1e-6,
    sensealg = ForwardDiffSensitivity()#SciMLSensitivity.DiffEqBase.SensitivityADPassThrough()
    )

control = ControlParameter(
    collect(0.0:0.1:12.0)[1:end-1], name = :fishing
)

# Single Shooting
layer = CorleoneCore.SingleShootingLayer(prob, Tsit5(),Int64[],[1], (control,))
ps, st = LuxCore.setup(Random.default_rng(), layer)
p = ComponentArray(ps)
sol, _ = layer(nothing, ps, st)

@code_warntype layer(nothing, ps, st);

using BenchmarkTools
#  635.042 μs (20747 allocations: 1.81 MiB)
@btime layer($nothing, $ps, $st);

loss = let layer = layer, st = st, ax = getaxes(p)
    (p, ::Any) -> begin
        ps = ComponentArray(p, ax)
        sols, _ = layer(nothing, ps, st)
        sols[:L][end] .+ 1e-3 * sum(abs2, sols[:w])
    end
end

Δzg = Zygote.gradient(Base.Fix2(loss, nothing), collect(p))[1]
Δfd = ForwardDiff.gradient(Base.Fix2(loss, nothing), collect(p))



loss(collect(p), nothing)

optfun = OptimizationFunction(
    loss, AutoZygote(),
)

lb, ub = copy(p), copy(p)
ub.controls .= 1.0

optprob = OptimizationProblem(
    optfun, collect(p), lb = collect(lb), ub = collect(ub)
)

uopt = solve(optprob, Ipopt.Optimizer(),
     tol = 1e-6,
     hessian_approximation = "limited-memory",
     max_iter = 300
)

optsol, _ = layer(nothing, uopt + zero(p), st)

f = Figure()
ax = CairoMakie.Axis(f[1,1])
ax1 = CairoMakie.Axis(f[2,1])
[plot!(ax, optsol.t, optsol[v]) for v in [:x, :y]]
[stairs!(ax1, optsol.t, optsol[v]) for v in [:w]]
f


## Multiple Shooting
shooting_points = [0.0, 3.0, 6.0, 12.0]
mslayer = CorleoneCore.MultipleShootingLayer(prob, Tsit5(),Int64[],[1], (control,), shooting_points)

msps, msst = LuxCore.setup(Random.default_rng(), mslayer)
# Or use any of the Initialization schemes
msps, msst = ConstantInitialization(Dict(1=>1.0,2=>1.0,3=>1.0))(Random.default_rng(), mslayer)
msps, msst = ForwardSolveInitialization()(Random.default_rng(), mslayer)
msp = ComponentArray(msps)


msloss = let layer = mslayer, st = msst, ax = getaxes(msp)
    (p, ::Any) -> begin
        ps = ComponentArray(p, ax)
        sols, _ = layer(nothing, ps, st)
        last(sols)[:L, end]
    end
end

shooting_constraints = let layer = mslayer, st = msst, ax = getaxes(msp)
    (p, ::Any) -> begin
        ps = ComponentArray(p, ax)
        sols, _ = layer(nothing, ps, st)
        reduce(vcat, map(zip(sols[1:end-1], keys(ax[1])[2:end])) do (sol, name_i)
            _u0 = getproperty(ps, name_i).u0
            sol[[:x, :y, :L]][end] .-_u0
        end)
    end
end

matching = shooting_constraints(msp, nothing)
eq_cons(res, x, p) = res .= shooting_constraints(x, p)

ms_lb = map(msps) do p_i
    p_i.controls .= 0.0
    p_i.u0 .= 0.05
    p_i
end |> ComponentArray


ms_ub = map(msps) do p_i
    p_i.controls .= 1.0
    p_i.u0 .= 10.0
    p_i
end |> ComponentArray


optfun = OptimizationFunction(
    msloss, AutoForwardDiff(), cons = eq_cons
)

optprob = OptimizationProblem(
    optfun, collect(msp), lb = collect(ms_lb), ub = collect(ms_ub), lcons = zero(matching), ucons=zero(matching)
)

uopt = solve(optprob, Ipopt.Optimizer(),
     tol = 1e-6,
     hessian_approximation = "limited-memory",
     max_iter = 300 # 165
)

blocks = CorleoneCore.get_block_structure(mslayer)

mssol, _ = mslayer(nothing, uopt + zero(msp), msst)

f = Figure()
ax = CairoMakie.Axis(f[1,1])
ax1 = CairoMakie.Axis(f[2,1])
ax2 = CairoMakie.Axis(f[3,1])
fobs = getsym(mssol[1], :((1-0.4*w)*x - log(x) + (1-0.2*w)*y - log(y)))
foreach(eachindex(mssol)) do i 
    sol = mssol[i]
    for (i, xi) in enumerate([:x, :y])
        scatterlines!(ax, sol[:t], sol[xi])
    end
    stairs!(ax1, sol[:t], sol[:w])

    lines!(ax2, sol[:t], fobs(sol))
end
f