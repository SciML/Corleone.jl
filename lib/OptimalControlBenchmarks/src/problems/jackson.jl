function jackson(grids)

    num_states = 3
    num_controls = 1
    tspan = (0.,1.)

    scaled_grids = scale_grids!(tspan, grids)

    @variables begin
        x₁(..) = 1., [tunable = false]
        x₂(..) = 0., [tunable = false]
        x₃(..) = 0., [tunable = false]
        u(..) = 0.5, [bounds = (0., 1.), input = true]
    end

    @constants begin
        k₁ = 1., [tunable = false]
        k₂ = 10., [tunable = false]
        k₃ = 1., [tunable = false]
    end

    eqs = [
        D(x₁(t)) ~ -u(t) * (k₁ * x₁(t) -k₂ * x₂(t))
        D(x₂(t)) ~ u(t) * (k₁ * x₁(t) - k₂ * x₂(t)) - (1 - u(t)) * k₃ * x₂(t)
        D(x₃(t)) ~ (1- u(t)) * k₃ * x₂(t)
    ]

    # Define control discretization
    constraint_grid = scaled_grids.constraint_grid

    grid_cons_u1 = [x₁(tᵢ) ≲ 1.1 for tᵢ in constraint_grid]
    grid_cons_u2 = [x₂(tᵢ) ≲ 1.1 for tᵢ in constraint_grid]
    grid_cons_u3 = [x₃(tᵢ) ≲ 1.1 for tᵢ in constraint_grid]
    grid_cons_l1 = [x₁(tᵢ) ≳ 0. for tᵢ in constraint_grid]
    grid_cons_l2 = [x₂(tᵢ) ≳ 0. for tᵢ in constraint_grid]
    grid_cons_l3 = [x₃(tᵢ) ≳ 0. for tᵢ in constraint_grid]

    cons = [
        grid_cons_u1..., grid_cons_u2..., grid_cons_u3...,
        grid_cons_l1..., grid_cons_l2..., grid_cons_l3...
    ]

    costs = [x₃(last(tspan))]

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
	name = "Jackson"
    )

end
