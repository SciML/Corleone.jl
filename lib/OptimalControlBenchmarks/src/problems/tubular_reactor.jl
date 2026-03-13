function tubular_reactor(grids)

    num_states = 2
    num_controls = 1
    tspan = (0.0, 1.0)

    scaled_grids = scale_grids!(tspan, grids)

    @variables begin
        x₁(..) = 1.0, [tunable = false]
        x₂(..) = 0.0, [tunable = false]
        u(..) = 0.0, [bounds = (0.0, 5.0), input = true]
    end

    eqs = [
        D(x₁(t)) ~ -(u(t) + 0.5 * u(t)^2) * x₁(t)
        D(x₂(t)) ~ u(t) * x₁(t)
    ]

    costs = [-x₂(last(tspan))]

    @named oc_problem = System(
        eqs,
        t;
        costs = costs
    )

    return (
        system = oc_problem,
        grids = scaled_grids,
        dims = (num_states, num_controls),
        name = "Tubular Reactor",
    )

end
