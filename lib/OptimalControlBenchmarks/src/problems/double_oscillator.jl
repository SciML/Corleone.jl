function double_oscillator(grids)

    num_states = 5
    num_controls = 1
    tspan = (0.0, 2 * pi)

    scaled_grids = scale_grids!(tspan, grids)

    @variables begin
        x₀(..) = 0.0, [tunable = false]
        x₁(..) = 0.0, [tunable = false]
        x₂(..) = 0.0, [tunable = true]
        x₃(..) = 0.0, [tunable = true]
        u(..) = 1.0, [bounds = (-1.0, 1.0), input = true]
    end

    @constants begin
        m₁ = 200.0, [tunable = false]
        m₂ = 2.0, [tunable = false]
        k₁ = 100.0, [tunable = false]
        k₂ = 3.0, [tunable = false]
        c = 0.5, [tunable = false]
        T = 2 * pi, [tunable = false]
    end

    eqs = [
        D(x₀(t)) ~ x₂(t)
        D(x₁(t)) ~ x₃(t)
        D(x₂(t)) ~ - (k₁ + k₂) / m₁ * x₀(t) + k₂ / m₁ * x₁(t) + 1 / m₁ * sin(2 * pi / T * t)
        D(x₃(t)) ~ k₂ / m₂ * x₀(t) - k₂ / m₂ * x₁(t) - c * (1 - u(t)) / m₂ * x₃(t)
    ]

    costs = [
        Symbolics.Integral(t in (0.0, 1.0))(
            0.5 * (x₀(t)^2 + x₁(t)^2 + u(t)^2)
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
        name = "Double Oscillator",
    )

end
