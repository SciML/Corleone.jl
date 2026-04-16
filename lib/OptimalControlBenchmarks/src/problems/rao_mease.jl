function rao_mease(grids)

    num_states = 2
    num_controls = 1
    tspan = (0.0, 10)

    scaled_grids = scale_grids!(tspan, grids)

    @variables begin
        x(..) = 1.0, [tunable = false]
        u(..) = 0.0, [input = true]
    end

    eqs = [
        D(x(t)) ~ -x(t)^3 + u(t),
    ]

    cons = [
        x(last(tspan)) ~ 1.5,
    ]

    costs = [
        Symbolics.Integral(t in (0.0, last(tspan)))(
            x(t)^2 + u(t)^2
        ),
    ]

    @named oc_problem = System(
        eqs,
        t;
        costs = costs,
        constraints = cons
    )

    return (
        system = oc_problem,
        grids = scaled_grids,
        dims = (num_states, num_controls),
        name = "Rao Mease",
    )

end
