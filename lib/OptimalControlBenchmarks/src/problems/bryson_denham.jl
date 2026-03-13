function bryson_denham(grids)

    num_states = 2
    num_controls = 1
    tspan = (0.,1.)

    scaled_grids = scale_grids!(tspan, grids)

    @variables begin
        x(..) = 0.0, [tunable = false]
        v(..) = 1.0, [tunable = false]
        w(..) = 0.0, [input = true]
    end

    eqs = [
        D(x(t)) ~ v(t)
        D(v(t)) ~ w(t)
    ]

    constraint_grid = scaled_grids.constraint_grid[1:end - 1]

    grid_cons = [x(tᵢ) ≲ 1/9 for tᵢ in constraint_grid]

    cons = [
        x(last(tspan)) ~ 0.,
        v(last(tspan)) ~ -1.,
        grid_cons...
    ]

    costs = [
        Symbolics.Integral(t in (0.0, 1.0))(w(t)^2)
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
	name = "Bryson Denham"
    )

end
