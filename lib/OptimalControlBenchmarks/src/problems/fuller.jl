function fuller(grids)

    num_states = 3
    num_controls = 1
    tspan = (0.0, 1.0)

    scaled_grids = scale_grids!(tspan, grids)

    @variables begin
        x₀(..) = 0.01, [tunable = false]
        x₁(..) = 0.0, [tunable = false]
        u(..) = 0.5, [bounds = (0.0, 1.0), input = true]
    end

    eqs = [
        D(x₀(t)) ~ x₁(t)
        D(x₁(t)) ~ 1.0 - 2.0 * u(t)
    ]

    cons = [
        x₀(last(tspan)) ~ 0.01,
        x₁(last(tspan)) ~ 0.0,
    ]

    costs = [
        Symbolics.Integral(t in (0.0, last(tspan)))(
            x₀(t)^2
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
        name = "Fuller's problem",
    )

end
