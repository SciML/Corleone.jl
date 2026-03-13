function batch_reactor(grids)

    num_states = 2
    num_controls = 1
    tspan = (0.,1.)

    scaled_grids = scale_grids!(tspan, grids)

    @variables begin
        x₁(..) = 1.0, [tunable = false]
        x₂(..) = 0.0, [tunable = false]
        u(..) = 298.0, [bounds = (298.0, 398.0), input = true]
    end

    eqs = [
        D(x₁(t)) ~ -4.e3 * exp(-2500 / u(t)) * x₁(t)^2
        D(x₂(t)) ~ 4.e3 * exp(-2500 / u(t)) * x₁(t)^2 - 62.e4 * exp(-5000 / u(t)) * x₂(t)^2
    ]

    costs = [-x₂(1.0)]

    @named oc_problem = System(
        eqs,
        t;
        costs = costs
    )

    return (
        system = oc_problem,
	grids = scaled_grids,
	dims = (num_states, num_controls),
	name = "Batch Reactor"
    )

end
