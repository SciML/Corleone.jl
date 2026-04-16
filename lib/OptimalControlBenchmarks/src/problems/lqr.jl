function lqr(grids)

    num_states = 2
    num_controls = 1
    tspan = (0.0, 10.0)

    scaled_grids = scale_grids!(tspan, grids)

    @variables begin
        x(..) = 1.0, [tunable = false]
        u(..) = 0.5, [input = true]
    end

    @constants begin
        a = -1.0, [tunable = false]
        b = 1.0, [tunable = false]
    end

    eqs = [
        D(x(t)) ~ a * x(t) + b * u(t),
    ]

    costs = [
        Symbolics.Integral(t in (0.0, last(tspan)))(
            10.0 * (x(t) - 3.0)^2 + 0.1 * u(t)^2
        ),
    ]

    @named oc_problem = System(
        eqs,
        t;
        costs = costs
    )

    return (
        system = oc_problem,
        grids = scaled_grids,
        dims = (num_states, num_controls),
        name = "Linear Quadratic Regulator",
    )

end
