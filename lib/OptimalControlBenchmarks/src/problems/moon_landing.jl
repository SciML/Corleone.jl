function moon_landing(grids)

    num_states = 3
    num_controls = 1
    tspan = (0.0, 1.0)

    scaled_grids = scale_grids!(tspan, grids)

    @variables begin
        h(..) = 1.0, [tunable = false]
        v(..) = -0.783, [tunable = false]
        m(..) = 1.0, [tunable = false]
        T(..) = 0.5, [bounds = (0.0, 1.227), input = true]
    end

    @parameters begin
        tₛ = 1.0, [bounds = (1.0e-3, Inf), tunable = true]
    end

    eqs = [
        D(h(t)) ~ tₛ * v(t)
        D(v(t)) ~ tₛ * (-1 + T(t) / m(t))
        D(m(t)) ~ tₛ * (- T(t) / 2.349)
    ]

    cons = [
        h(last(tspan)) ~ 0.0,
        v(last(tspan)) ~ 0.0,
    ]

    costs = [-m(last(tspan))]

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
        name = "Moon Landing",
    )

end
