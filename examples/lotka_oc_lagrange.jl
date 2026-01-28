using Corleone
using OrdinaryDiffEq
using SciMLSensitivity
using ComponentArrays
using LuxCore
using Random

using CairoMakie
using Optimization
using OptimizationMOI

function lotka_dynamics(u, p, t)
    return [u[1] - p[2] * prod(u[1:2]) - 0.4 * p[1] * u[1];
            -u[2] + p[3] * prod(u[1:2]) - 0.2 * p[1] * u[2];
            (u[1]-1.0)^2 + (u[2] - 1.0)^2]
end

tspan = (0., 12.)
u0 = [0.5, 0.7, 0.]
p0 = [0.0, 1.0, 1.0]

lotka_dynamics(u0, p0, tspan[1])

prob = ODEProblem(lotka_dynamics, u0, tspan, p0)


N_stages = 40
N_controls = 120

control = ControlParameter(
    range(0.,12.,N_controls + 1), name = :fishing, bounds=(0.0,1.0)
)

shooting_points = [range(0.,12.,N_stages + 1)...]
mslayer = MultipleShootingLayer(prob, Tsit5(), shooting_points...; controls = (1 => control,),
                            bounds_p = ([1.0, 1.0], [1.0, 1.0]), 
                            quadrature_indices = 3:3
                            )

msps, msst = LuxCore.setup(Random.default_rng(), mslayer)
msp = ComponentArray(msps)
ms_lb, ms_ub = Corleone.get_bounds(mslayer) .|> ComponentArray

function stage_set_all!(MSCARR::ComponentArray, inner_ind, val)
    for interval in keys(MSCARR)
        view(MSCARR, interval)[inner_ind] .= val[1:length(view(MSCARR, interval)[inner_ind])]
    end
end
stage_set_all!(ms_lb, :u0, [0., 0., -Inf])

msloss = let layer = mslayer, st = msst, ax = getaxes(msp)
    (p, ::Any) -> begin
        ps = ComponentArray(p, ax)
        sols, _ = layer(nothing, ps, st)
        last(sols.u)[3]
    end
end

shooting_constraints = let layer = mslayer, st = msst, ax = getaxes(msp)
    (res, p, ::Any) -> begin
        ps = ComponentArray(p, ax)
        sols, _ = layer(nothing, ps, st)
        Corleone.shooting_constraints!(res, sols)
    end
end

const N_match::Int64 = Corleone.get_number_of_shooting_constraints(mslayer, msps, msst)
matching_fun(p::AbstractVector{T}) where T = (res = Vector{T}(undef, N_match); shooting_constraints(res, p, nothing); res)

matching = shooting_constraints(zeros(Corleone.get_number_of_shooting_constraints(mslayer, msps, msst)), msp, nothing)

optfun = OptimizationFunction(
    msloss, AutoForwardDiff(), cons = shooting_constraints
)


optprob = OptimizationProblem(
    optfun, collect(msp), lb = collect(ms_lb), ub = collect(ms_ub), lcons = zero(matching), ucons=zero(matching)
)

using Ipopt
uopt = solve(optprob, Ipopt.Optimizer(),
     tol = 1e-6,
     hessian_approximation = "limited-memory",
     max_iter = 300
)


blocks = Corleone.get_block_structure(mslayer)

using blockSQP
opt_BSQP_sparse = blockSQP.sparse_options()
# Activate adaptive temination
opt_BSQP_sparse.enable_premature_termination = true
opt_BSQP_sparse.max_extra_steps = 10

uopt = solve(optprob, BlockSQPOpt(),
    opttol = 1e-6,
    options = opt_BSQP_sparse,
    sparsity = blocks,
    maxiters = 300
)

mssol, _ = mslayer(nothing, uopt + zero(msp), msst)

f = Figure()
ax = CairoMakie.Axis(f[1,1])
scatterlines!(ax, mssol, vars=[:x₁, :x₂])
f[1, 2] = Legend(f, ax, "States", framevisible = false)
ax1 = CairoMakie.Axis(f[2,1])
stairs!(ax1, mssol, vars=[:u₁])
f[2, 2] = Legend(f, ax1, "Controls", framevisible = false)
display(f)

