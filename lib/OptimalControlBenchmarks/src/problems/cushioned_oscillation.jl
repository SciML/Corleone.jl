function cushioned_oscillation(grids)

    num_states = 3
    num_controls = 1
    tspan = (0.0, 10.0)

    scaled_grids = scale_grids!(tspan, grids)

    @variables begin
        x(..) = 2.0, [tunable = false]
        v(..) = 5.0, [tunable = false]
        obj(..) = 0.0, [tunable = false]
        u(..) = 0.0, [bounds = (-5.0, 5.0), input = true]
    end
    @parameters begin
        tₛ = 1.0, [bounds = (1.0e-3, Inf), tunable = true]
    end
    @constants begin
        m = 5.0, [tunable = false]
        c = 10.0, [tunable = false]
    end

    eqs = [
        D(x(t)) ~ tₛ * v(t)
        D(v(t)) ~ tₛ * (1 / m * (u(t) - c * x(t)))
        D(obj(t)) ~ tₛ
    ]

    cons = [
        x(last(tspan)) ~ 0.0,
        v(last(tspan)) ~ 0.0,
    ]

    costs = [obj(1.0)]

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
        name = "Cushioned Oscillation",
    )

end
