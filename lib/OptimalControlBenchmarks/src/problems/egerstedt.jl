function egerstedt(grids)

    num_states = 3
    num_controls = 3
    tspan = (0.,1.)

    scaled_grids = scale_grids!(tspan, grids)

    @variables begin
        x(..) = 0.5, [tunable = false]
        y(..) = 0.5, [tunable = false]
        u_1(..) = 1 / 3, [bounds = (-1.0, 1.0), input = true]
        u_2(..) = 1 / 3, [bounds = (-1.0, 1.0), input = true]
        u_3(..) = 1 / 3, [bounds = (-1.0, 1.0), input = true]
    end

    eqs = [
        D(x(t)) ~ -x(t) * u_1(t) + (x(t) + y(t)) * u_2(t) + (x(t) - y(t)) * u_3(t)
        D(y(t)) ~ (x(t) + 2 * y(t)) * u_1(t) + (x(t) - 2 * y(t)) * u_2(t) + (x(t) + y(t)) * u_3(t)
    ]

    # scale the constraint grid
    constraint_grid = scaled_grids.constraint_grid

    grid_cons_x = [y(tᵢ) ≳ 0.4 for tᵢ in constraint_grid]
    grid_cons_u = [u_1(tᵢ) + u_2(tᵢ) + u_3(tᵢ) ~ 1. for tᵢ in constraint_grid]

    cons = [
        grid_cons_x...,
        grid_cons_u...
    ]

    costs = [
        Symbolics.Integral(t in (0.0, 1.0))(
            x(t)^2 + y(t)^2
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
	dims = (num_states, num_controls)
    )

end
