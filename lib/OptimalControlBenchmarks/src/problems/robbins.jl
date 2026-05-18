function robbins(grids)

    num_states = 4
    num_controls = 1
    tspan = (0.0, 10.0)

    scaled_grids = scale_grids!(tspan, grids)

    @variables begin
        x₁(..) = 1.0, [tunable = false]
        x₂(..) = -2.0, [tunable = false]
        x₃(..) = 0.0, [tunable = false]
        u(..) = 0.0, [input = true]
    end

    @constants begin
        α = 3.0, [tunable = false]
        β = 0.0, [tunable = false]
        γ = 0.5, [tunable = false]
    end

    eqs = [
        D(x₁(t)) ~ x₂(t)
        D(x₂(t)) ~ x₃(t)
        D(x₃(t)) ~ u(t)
    ]

    # scale the constraint grid
    constraint_grid = scaled_grids.constraint_grid[1:(end - 1)]

    grid_cons_x₁ = [x₃(tᵢ) ≳ 0.0 for tᵢ in constraint_grid]

    cons = [
        x₁(last(tspan)) ~ 0.0,
        x₂(last(tspan)) ~ 0.0,
        x₃(last(tspan)) ~ 0.0,
        grid_cons_x₁...,
    ]

    costs = [
        Symbolics.Integral(t in (0.0, last(tspan)))(
            α * x₁(t) + β * x₁(t)^2 + γ * u(t)^2
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
        name = "Robbins",
    )

end
