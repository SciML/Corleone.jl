using Corleone
using OrdinaryDiffEq
using SciMLSensitivity
using LuxCore
using Random
rng = Random.default_rng()
using ComponentArrays

using Optimization
using OptimizationMOI
using Ipopt
using blockSQP
using CairoMakie

using UnPack


params = Dict(
    [
        :x0 => 0,
        :y0 => 1000,
        :ytf => 900,
        :dxbc => 13.23,
        :dybc => -1.288,
        :c0 => 0.034,
        :c1 => 0.069662,
        :S => 14.0,
        :rho => 1.13,
        :cmax => 1.4,
        :m => 100,
        :g => 9.81,
        :uC => 2.5,
        :rC => 100,
    ]
)

function gliderRHS(u, p, t)
    x, dx, y, dy = u
    cL, T = p
    @unpack c0, c1, S, rho, m, g, uC, rC = params
    r = (x / rC - 2.5)^2
    u = uC * (1 - r) * exp(-r)
    w = dy - u
    v = sqrt(dx^2 + w^2)
    D = 1 / 2 * (c0 + c1 * cL^2) * rho * S * v^2
    L = 1 / 2 * cL * rho * S * v^2
    return [
        dx;
        1 / m * (-L * w / v - D * dx / v);
        dy;
        1 / m * (L * dx / v - D * w / v) - g
    ] .* T
end

@unpack x0, dxbc, y0, dybc, ytf, cmax = params
tspan = (0.0, 1.0)
u0 = [x0, dxbc, y0, dybc]
p0 = [cmax, 100.0]

gliderRHS(u0, p0, tspan[1])

ODE = ODEProblem(gliderRHS, u0, tspan, p0)

C_lift = ControlParameter(collect(range(tspan[1], tspan[2], 51)), name = :lift, bounds = (0.0, cmax))

layer = Corleone.SingleShootingLayer(ODE, Tsit5(), controls = [(1, C_lift)])
ps, st = LuxCore.setup(rng, layer)

p = ComponentArray(ps)
lb, ub = ComponentArray.(Corleone.get_bounds(layer))

lb.p .= 0.1
ub.p .= 100

sols, _ = layer(nothing, p, st)

obj = let layer = layer, st = st, ax = getaxes(p)
    (p_arr, ::Any) -> begin
        p_carr = ComponentArray(p_arr, ax)
        sols, _ = layer(nothing, p_carr, st)
        -sols[:x₁][end]
    end
end

obj(collect(p), nothing)

_term_cons = let layer = layer, st = st, ax = getaxes(p)
    (p_arr, ::Any) -> begin
        p_carr = ComponentArray(p_arr, ax)
        sols, _ = layer(nothing, p_carr, st)
        sols.u[end][2:4] - [dxbc, ytf, dybc]
    end
end

term_cons(res, arg_u, arg_p) = res .= _term_cons(arg_u, arg_p)

optfcn = OptimizationFunction(obj, AutoForwardDiff(); cons = term_cons) #Why does AutoZygote not work?
optprob = OptimizationProblem(
    optfcn, collect(p);
    lb = collect(lb), ub = collect(ub),
    lcons = zeros(3), ucons = zeros(3)
)

uopt = solve(
    optprob, Ipopt.Optimizer(),
    tol = 1.0e-6,
    hessian_approximation = "limited-memory",
    max_iter = 300
)
optsol, _ = layer(nothing, ComponentArray(uopt, getaxes(p)), st)

Tgrid_opt = optsol.t * optsol[:p₁]

f = Figure()
ax = CairoMakie.Axis(f[1, 1])
lines!(ax, Tgrid_opt, optsol[:x₁], label = "x", color = :skyblue3)
lines!(ax, Tgrid_opt, (optsol[:x₃] .- 900) .* 10, color = :deepskyblue, label = "(y - 900)⋅10")

lines!(ax, Tgrid_opt, optsol[:x₂] .* 100, linestyle = :dashdot, color = :deepskyblue3, label = "∂x/∂t")
lines!(ax, Tgrid_opt, optsol[:x₄] .* 100, linestyle = :dashdot, color = :deepskyblue, label = "∂y/∂t")

f[1, 2] = Legend(f, ax, "States", framevisible = false)
ax1 = CairoMakie.Axis(f[2, 1])
stairs!(ax1, Tgrid_opt, optsol[:u₁], color = :deepskyblue2, label = "cL")
f[2, 2] = Legend(f, ax1, "Controls", framevisible = false)
display(f)
