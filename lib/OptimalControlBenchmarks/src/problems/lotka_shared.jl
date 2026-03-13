function lotka_shared(grids)

    num_states = 4
    num_controls = 1
    tspan = (0.,40.)

    scaled_grids = scale_grids!(tspan, grids)

    @variables begin
        x₀(..) = 1.5, [tunable = false]
        x₁(..) = 1., [tunable = false]
        x₂(..) = 0.5, [tunable = false]
        u(..) = 0.5, [bounds = (0., 1.), input = true]
    end

    @constants begin
        c₁ = 0.1, [tunable = false]
        c₂ = 0.4, [tunable = false]
        α = 1.2, [tunable = false]
    end

    eqs = [
        D(x₀(t)) ~ x₀(t) - x₀(t) * x₁(t) - x₀(t) * x₂(t)
        D(x₁(t)) ~ -x₁(t) + x₀(t) * x₁(t) - c₁ * x₁(t) * u(t)
        D(x₂(t)) ~ -x₂(t) + α * x₀(t) * x₂(t) - c₂ * x₂(t) * u(t)
    ]

    costs = [
        Symbolics.Integral(t in (0.0, last(tspan)))(
            (x₀(t) - 1.5)^2 + (x₁(t) - 1.)^2 + (x₂(t) - 1.)^2
        ),
    ]

    @named oc_problem = System(
        eqs,
        t;
        costs = costs
    )

    return (
        system = oc_problem,
	grids = scaled_grids,
	dims = (num_states, num_controls),
	name = "Lotka Volterra Shared"
    )

end
