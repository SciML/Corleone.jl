function lotka_volterra(grids)

    num_states = 3
    num_controls = 1
    tspan = (0.0, 12.0)

    scaled_grids = scale_grids!(tspan, grids)

    @variables begin
        x₀(..) = 0.5, [tunable = false]
        x₁(..) = 0.7, [tunable = false]
        u(..) = 0.5, [bounds = (0.0, 1.0), input = true]
    end

    @constants begin
        c₀ = 0.4, [tunable = false]
        c₁ = 0.2, [tunable = false]
    end

    eqs = [
        D(x₀(t)) ~ x₀(t) - x₀(t) * x₁(t) - c₀ * x₀(t) * u(t)
        D(x₁(t)) ~ -x₁(t) + x₀(t) * x₁(t) - c₁ * x₁(t) * u(t)
    ]

    costs = [
        Symbolics.Integral(t in (0.0, last(tspan)))(
            (x₀(t) - 1.0)^2 + (x₁(t) - 1.0)^2
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
        name = "Lotka Volterra fishing problem",
    )

end
