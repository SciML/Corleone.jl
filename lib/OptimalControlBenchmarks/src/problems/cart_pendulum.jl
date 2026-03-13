function cart_pendulum(grids)

    num_states = 5
    num_controls = 1
    tspan = (0.0, 4.0)

    scaled_grids = scale_grids!(tspan, grids)

    @variables begin
        x(..) = 0.0, [tunable = false]
        θ(..) = 0.0, [tunable = false]
        dx(..) = 0.0, [tunable = false]
        dtheta(..) = 0.0, [tunable = false]
        w(..) = 0.0, [bounds = (-30, 30), input = true]
    end

    @parameters begin
        α = 10.0, [tunable = false]
        β = 50.0, [tunable = false]
        γ = 0.5, [tunable = false]
        M = 1.0, [tunable = false]
        m = 0.1, [tunable = false]
        g = 9.81, [tunable = false]
    end

    eqs = [
        D(x(t)) ~ dx(t)
        D(θ(t)) ~ dtheta(t)
        D(dx(t)) ~ (w(t) + m * g * sin(θ(t)) * cos(θ(t)) + m * dtheta(t)^2 * sin(θ(t))) / (M + m * (1 - cos(θ(t)))^2)
        D(dtheta(t)) ~ -g * sin(θ(t)) - ((w(t) + m * g * sin(θ(t)) * cos(θ(t)) + m * dtheta(t)^2 * sin(θ(t))) / (M + m * (1 - cos(θ(t))^2))) * cos(θ(t))
    ]

    constraint_grid = scaled_grids.constraint_grid

    grid_cons_le = [x(tᵢ) ≲ 2.0 for tᵢ in constraint_grid]
    grid_cons_ge = [x(tᵢ) ≳ -2.0 for tᵢ in constraint_grid]

    cons = [
        grid_cons_le...,
        grid_cons_ge...,
    ]

    costs = [
        Symbolics.Integral(t in (0.0, 1.0))(
            1.0e-3 * (α * x(t)^2 + β * (θ(t) - pi)^2 + γ * w(t)^2)
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
        name = "Cart Pendulum",
    )

end
