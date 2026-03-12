function quadrotor(grids)

    num_states = 7
    num_controls = 4
    tspan = (0.,7.5)

    scaled_grids = scale_grids!(tspan, grids)
    
    @variables begin
        x₁(..) = 0., [tunable = false]
        x₂(..) = 0., [tunable = false]
        x₃(..) = 1., [tunable = false]
        x₄(..) = 0., [tunable = false]
        x₅(..) = 0., [tunable = false]
        x₆(..) = 0., [tunable = false]
        w₁(..) = 1. / 3., [bounds = (0., 1.), input = true]
        w₂(..) = 1. / 3., [bounds = (0., 1.), input = true]
        w₃(..) = 1. / 3., [bounds = (0., 1.), input = true]
        u(..) = 0., [bounds = (0., 1.e-3), input = true]
    end
    
    @constants begin
        g = 9.81, [tunable = false]
        M = 1.3, [tunable = false]
        L = 0.305, [tunable = false]
        I = 0.0605, [tunable = false]	
    end
    
    eqs = [
        D(x₁(t)) ~ x₂(t)
        D(x₂(t)) ~ g * sin(x₅(t)) + w₁(t) * u(t) * sin(x₅(t)) / M
        D(x₃(t)) ~ x₄(t)
        D(x₄(t)) ~ g * cos(x₅(t)) - g + w₁(t) * u(t) * cos(x₅(t)) / M
        D(x₅(t)) ~ x₆(t)
        D(x₆(t)) ~ -w₂(t) * L * u(t) / I + w₃(t) * L * u(t) / I
    ]
    
    constraint_grid = scaled_grids.constraint_grid   
    grid_cons_w = [w₁(tᵢ) + w₂(tᵢ) + w₃(tᵢ) ~ 1. for tᵢ in constraint_grid]
    grid_cons_x₃ = [x₃(tᵢ) ≳ 0. for tᵢ in constraint_grid]
    
    cons = [
        grid_cons_w...,
        grid_cons_x₃...
    ]
    
    costs = [
        5 * (x₁(t_f) - 6)^2 + 5 * (x₃(t_f) - 1)^2 + 5 * (sin(x₅(t_f) * 0.5))^2 + Symbolics.Integral(t in (0.0, last(tspan)))(
            5 * u(t)^2
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
