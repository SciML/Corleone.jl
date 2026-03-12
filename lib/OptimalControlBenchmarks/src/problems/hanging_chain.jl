function hanging_chain(grids)

    num_states = 3
    num_controls = 1
    tspan = (0.,1.)

    scaled_grids = scale_grids!(tspan, grids)

    @variables begin
        x₁(..) = 1., [tunable = false]
        x₂(..) = 0., [tunable = false]
        x₃(..) = 0., [tunable = false]
        u(..) = 0.5, [bounds = (-10., 20.), input = true]
    end

    @constants begin
        a = 1., [tunable = false]
        b = 3., [tunable = false]
        L_p = 4., [tunable = false]
    end

    eqs = [
        D(x₁(t)) ~ u(t)
        D(x₂(t)) ~ x₁(t) * sqrt(1 + u(t)^2)
        D(x₃(t)) ~ sqrt(1 + u(t)^2)
    ]

    # scale the constraint grid
    constraint_grid = scaled_grids.constraint_grid

    grid_cons_u1 = [x₁(tᵢ) ≳ 0. for tᵢ in constraint_grid[1:end - 1]]
    grid_cons_u2 = [x₂(tᵢ) ≳ 0. for tᵢ in constraint_grid]
    grid_cons_u3 = [x₃(tᵢ) ≳ 0. for tᵢ in constraint_grid[1:end - 1]]
    grid_cons_l1 = [x₁(tᵢ) ≲ 10. for tᵢ in constraint_grid[1:end - 1]]
    grid_cons_l2 = [x₂(tᵢ) ≲ 10. for tᵢ in constraint_grid]
    grid_cons_l3 = [x₃(tᵢ) ≲ 10. for tᵢ in constraint_grid[1:end - 1]]

    cons = [
        x₁(last(tspan)) - 3. ~ 0.,
        x₃(last(tspan)) - 4. ~ 0.,
        grid_cons_u1..., grid_cons_u2..., grid_cons_u3...,
        grid_cons_l1..., grid_cons_l2..., grid_cons_l3...
    ]

    costs = [x₂(last(tspan))]

    @named oc_problem = System(
        eqs,
        t;
        costs = costs,
        constraints = cons
    )

    return (
        system = oc_problem,
	grids = scaled_grids,
	dims = (num_states, num_controls)
    )

end
