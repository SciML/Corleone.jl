function van_der_pol(grids)

    num_states = 3
    num_controls = 1
    tspan = (0.0, 10.0)

    scaled_grids = scale_grids!(tspan, grids)

    @variables begin
        x₁(..) = 0.0, [tunable = false]
        x₂(..) = 1.0, [tunable = false]
        u(..) = 0.0, [bounds = (-1.0, 1.0), input = true]
    end

    eqs = [
        D(x₁(t)) ~ (1 - x₂(t)^2) * x₁(t) - x₂(t) + u(t)
        D(x₂(t)) ~ x₁(t)
    ]

    constraint_grid = scaled_grids.constraint_grid

    grid_cons_x₁ = [x₁(tᵢ) ≳ -0.25 for tᵢ in constraint_grid]

    cons = [
        grid_cons_x₁...,
    ]

    costs = [
        Symbolics.Integral(t in (0.0, last(tspan)))(
            x₁(t)^2 + x₂(t)^2 + u(t)^2
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
        name = "Van der Pol Oscillator",
    )

end
