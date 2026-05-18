function dielectrophoretic_particle(grids)

    num_states = 3
    num_controls = 1
    tspan = (0.0, 5.0)

    scaled_grids = scale_grids!(tspan, grids)

    @variables begin
        x₀(..) = 1.0, [tunable = false]
        x₁(..) = 0.0, [tunable = false]
        obj(..) = 0.0, [tunable = false]
        u(..) = 1.0, [bounds = (-1.0, 1.0), input = true]
    end
    @parameters begin
        tₛ = 1.0, [bounds = (1.0e-1, Inf), tunable = true]
    end
    @constants begin
        α = -0.75, [tunable = false]
        c = 1.0, [tunable = false]
    end

    eqs = [
        D(x₀(t)) ~ tₛ * (x₁(t) * u(t) + α * u(t)^2)
        D(x₁(t)) ~ tₛ * (-c * x₁(t) + u(t))
        D(obj(t)) ~ tₛ
    ]

    cons = [
        x₀(last(tspan)) ~ 2.0,
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
        name = "Dielectrophoretic Particle",
    )

end
