# ---
# title: Minimal-time Lotka–Volterra problem
# description: Steer the fishery to its steady state in minimal time on a normalized horizon
# id: lotka_min_time
# julia: "1.10"
# notebook: false
# generate_cover: false
# ---

# This example steers the
# [Lotka–Volterra fishing problem](https://mintoc.de/index.php?title=Lotka_Volterra_fishing_problem)
# from its initial state ``(x, y) = (0.5, 0.7)`` to the steady state ``(1, 1)`` in
# **minimal time**, with the fishing intensity bounded to ``0 \le v \le 1``.
#
# The physical time horizon ``T`` is unknown. With the standard time
# transformation ``t = T\tau`` on the normalized time ``\tau \in [0, 1]``, the
# right-hand sides are scaled by ``T``, and a third state accumulates the elapsed
# physical time, ``\dot{t} = T``. The objective is its terminal value ``T``.
#
# !!! tip "Free end times in Corleone"
#     Corleone integrates on a fixed time span. Free horizons are handled by a
#     tunable scaling parameter, exactly as shown here.

# ## [Setup](@id lotka_min_time_setup)

using Corleone

using LuxCore
using Random
using OrdinaryDiffEqTsit5
using SymbolicIndexingInterface
using Optimization
using OptimizationMOI
using Ipopt
using ComponentArrays
using ForwardDiff
using CairoMakie
Makie.inline!()

# ## Dynamics in Normalized Time

# The parameter vector is `p = [T, v, c₁, c₂]`: the horizon, the fishing control,
# and the two fixed catch coefficients `c₁ = 0.4` and `c₂ = 0.2`.
#
# ```math
# \begin{aligned}
# \dot{x} &= T(x-xy-c_1vx), & x(0)&=0.5, \\
# \dot{y} &= T(-y+xy-c_2vy), & y(0)&=0.7, \\
# \dot{t} &= T, & t(0)&=0, & 0&\le v\le1.
# \end{aligned}
# ```

function lotka_scaled(du, u, p, t)
    x, y, time = u
    T, v, c₁, c₂ = p
    du[1] = T * (x - x * y - c₁ * v * x)
    du[2] = T * (-y + x * y - c₂ * v * y)
    du[3] = T
    return nothing
end

prob = ODEProblem(
    lotka_scaled, [0.5, 0.7, 0.0], (0.0, 1.0), [12.0, 0.0, 0.4, 0.2];
    abstol = 1.0e-9, reltol = 1.0e-9
)

# ## Single Shooting Approach

# The horizon ``T`` is a single tunable value (no switching times), the fishing
# intensity is piecewise constant on 40 intervals of the normalized horizon, and
# the catch coefficients are `Constant`.
cgrid = collect(0.025:0.025:0.975)
layer = ShootingLayer(
    prob, Tsit5(),
    Tunable(
        1; kind = Val(:parameter), default = 12.0, lowerbound = 1.0, upperbound = 30.0
    ) => nothing,
    Tunable(
        2; kind = Val(:parameter), default = 0.5, lowerbound = 0.0, upperbound = 1.0
    ) => cgrid,
    Constant([3, 4]; kind = Val(:parameter)) => nothing;
    controls = [2],
)

# Minimize the elapsed time and steer the populations exactly to the steady state
# ``(1, 1)`` with two terminal equality constraints.
dynprob = CorleoneDynamicOptProblem(
    layer, :(u₃(1.0)),
    :(u₁(1.0) - 1.0),
    :(u₂(1.0) - 1.0);
    lcons = [0.0, 0.0], ucons = [0.0, 0.0]
)
opt = OptimizationProblem(dynprob, AutoForwardDiff(); function_kwargs = (;))
uopt = solve(
    opt, Ipopt.Optimizer();
    tol = 1.0e-8, hessian_approximation = "limited-memory",
    max_iter = 1000, print_level = 0
)
@assert successful_retcode(uopt)

# Reconstruct the optimized parameters. The trajectory stores every parameter as
# a time series, so the horizon and the control can be read by symbol.
_, reconstruct = pack(default_vectorizer(), dynprob.ps)
optimized_parameters = reconstruct(uopt.u)
optsol, _ = layer(nothing, optimized_parameters, opt.p)
T = first(optsol[:p₁])
v = Vector(optimized_parameters.parameter.parameters.p₂.parameter.ps)
(;
    minimal_time = T,
    terminal_state = round.(last(optsol.u); digits = 6),
    active_intervals = "$(count(>(0.01), v)) / $(length(v))",
)

# ## Plotting in Physical Time

# Map the normalized time back to physical time and resample the control on the
# dense trajectory time for a stair plot.
t_dense = T .* optsol.t
v_full = v[clamp.(searchsortedlast.(Ref(cgrid), optsol.t), 1, length(v))]

function plot_min_time(t_dense, u, v_full)
    f = Figure(size = (1000, 750))
    c = Makie.wong_colors()
    ax1 = CairoMakie.Axis(
        f[1, 1], title = "Populations", xlabel = "t", ylabel = "population"
    )
    lines!(ax1, t_dense, u[1, :]; linewidth = 2, label = "x (prey)", color = c[1])
    lines!(ax1, t_dense, u[2, :]; linewidth = 2, label = "y (predator)", color = c[2])
    hlines!(ax1, [1.0]; color = :gray, linestyle = :dot)
    axislegend(ax1; position = :rt)

    axc = CairoMakie.Axis(
        f[2, 1], title = "Optimal fishing control", xlabel = "t", ylabel = "v(t)"
    )
    stairs!(axc, t_dense, v_full; linewidth = 2, step = :pre)
    return f
end

plot_min_time(t_dense, reduce(hcat, optsol.u), v_full)
