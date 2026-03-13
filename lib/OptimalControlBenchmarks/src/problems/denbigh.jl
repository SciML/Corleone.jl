function denbigh(grids)

    # We start by defining our system
    num_states = 3
    num_controls = 1
    tspan = (0.0, 1000.0)

    scaled_grids = scale_grids!(tspan, grids)

    @variables begin
        x₁(..) = 1.0, [tunable = false]
        x₂(..) = 0.0, [tunable = false]
        x₃(..) = 0.0, [tunable = false]
        T(..) = 300.0, [bounds = (273.0, 415.0), input = true]
    end

    @parameters begin
        tₛ = 1.0, [bounds = (1.0e-3, Inf), tunable = true]
    end

    @constants begin
        E[1:4] = [3.0e3, 6.0e3, 3.0e3, 0.0], [tunable = false]
        k⁰[1:4] = [1.0e3, 1.0e7, 1.0e1, 1.0e-3], [tunable = false]
    end

    # auxiliary equations for the kᵢ
    k = [k⁰[i] * exp(-E[i] / T(t)) for i in [1, 2, 3, 4]]

    eqs = [
        D(x₁(t)) ~ -k[1] * x₁(t) - k[2] * x₁(t)
        D(x₂(t)) ~ k[1] * x₁(t) - k[3] + k[4] * x₂(t)
        D(x₃(t)) ~ k[3] * x₂(t)
    ]

    costs = [-x₃(last(tspan))]

    @named oc_problem = System(
        eqs,
        t;
        costs = costs
    )

    return (
        system = oc_problem,
        grids = scaled_grids,
        dims = (num_states, num_controls),
        name = "Denbigh Reaction",
    )

end
