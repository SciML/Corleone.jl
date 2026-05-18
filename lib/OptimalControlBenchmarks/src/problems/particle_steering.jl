function particle_steering(grids)

    num_states = 5
    num_controls = 1
    tspan = (0.0, 10.0)

    scaled_grids = scale_grids!(tspan, grids)

    @variables begin
        x₁(..) = 0.0, [tunable = false]
        x₂(..) = 0.0, [tunable = false]
        dx₁(..) = 0.0, [tunable = false]
        dx₂(..) = 0.0, [tunable = false]
        obj(..) = 0.0, [tunable = false]
        u(..) = 0.0, [bounds = (-pi / 2.0, pi / 2.0), input = true]
    end

    @parameters begin
        tₛ = 3.0e-2, [bounds = (1.0e-3, Inf), tunable = true]
    end

    @constants begin
        a = 100.0, [tunable = false]
    end

    eqs = [
        D(x₁(t)) ~ tₛ * (dx₁(t))
        D(x₂(t)) ~ tₛ * (dx₂(t))
        D(dx₁(t)) ~ tₛ * (a * cos(u(t)))
        D(dx₂(t)) ~ tₛ * (a * sin(u(t)))
        D(obj(t)) ~ tₛ
    ]

    cons = [
        x₂(last(tspan)) ~ 5.0,
        dx₁(last(tspan)) ~ 45.0,
        dx₂(last(tspan)) ~ 0.0,
    ]

    costs = [
        obj(last(tspan)),
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
        name = "Particle steering problem",
    )

end
