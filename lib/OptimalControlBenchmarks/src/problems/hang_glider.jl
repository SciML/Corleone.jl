module hang_glider

using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using Symbolics
using ..OptimalControlBenchmarks: OptimalControlBenchmark

function make_problem(constraint_grid=collect(0.:0.1:1.))

    num_states = 4
    num_controls = 1
    tspan = (0.,100.)

    @variables begin
        x(..) = 0., [tunable = false]
        y(..) = 1.e3, [tunable = false]
        v_x(..) = 13.23, [tunable = false]
        v_y(..) = -1.288, [tunable = false]
        c_L(..) = 0.5, [bounds = (0., 1.4), input = true]
    end

    @parameters begin
        T = 0.5, [bounds =(1.e-3, Inf), tunable = true]
    end

    @constants begin
        u_c = 2.5, [tunable = false]
        r_c = 100., [tunable = false]
        c_0 = 0.034, [tunable = false]
        c_1 = 0.069662, [tunable = false]
        S = 14., [tunable = false]
        rho = 1.13, [tunable = false]
        m = 100., [tunable = false]
        g = 9.81, [tunable = false]
    end

    # auxiliary functions
    r = (x(t) / r_c - 2.5)^2
    U_up = u_c * (1 - r) * exp(-r)
    w = v_y(t) - U_up
    v = sqrt(v_x(t)^2 + w^2)
    Dr = 0.5 * rho * S * (c_0 + c_1 * c_L(t)^2) * v^2
    L = 0.5 * rho * S * c_L(t) * v^2

    eqs = [
        D(x(t)) ~ T * v_x(t)
        D(y(t)) ~ T * v_y(t)
        D(v_x(t)) ~ T * (-1.) * (L * w + Dr * v_x(t)) / (m * v)
        D(v_y(t)) ~ T * ((L * v_x(t) - Dr * w) / (m * v) - g)
    ]

    # scale the constraint grid
    constraint_grid = constraint_grid * (last(tspan) - first(tspan))
    constraint_grid = (constraint_grid .+ first(tspan))[1:end - 1]

    grid_cons_x = [x(tᵢ) ≳ 0. for tᵢ in constraint_grid]
    grid_cons_vx = [v_x(tᵢ) ≳ 0. for tᵢ in constraint_grid]

    cons = [
        y(last(tspan)) ~ 900.,
        v_x(last(tspan)) ~ 13.23,
        v_y(last(tspan)) ~ -1.288,
        grid_cons_x...,
        grid_cons_vx...
    ]

    costs = [-x(last(tspan))]

    @named oc_problem = System(
        eqs,
        t;
        costs = costs,
        constraints = cons
    )

    return (
        system = oc_problem,
        tspan = tspan,
        num_states = num_states,
        num_controls = num_controls
    )

end


benchmark = OptimalControlBenchmark(
    :hang_glider,
    "Double integrator with quadratic control cost",
    make_problem
)

end