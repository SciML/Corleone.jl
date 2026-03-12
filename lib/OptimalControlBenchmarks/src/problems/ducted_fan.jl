function ducted_fan(grids)

    num_states = 7
    num_controls = 2
    tspan = (0.,10.)

    scaled_grids = scale_grids!(tspan, grids)
    
    @variables begin
        x₁(..) = 0.0, [tunable = false]
        v₁(..) = 0.0, [tunable = false]
        x₂(..) = 0.0, [tunable = false]
        v₂(..) = 0.0, [tunable = false]
        α(..) = 0.0, [tunable = false]
        vα(..) = 0.0, [tunable = false]
        u₁(..) = 0.0, [bounds = (-5.0, 5.0), input = true]
        u₂(..) = 8.5, [bounds = (0.0, 17.0), input = true]
    end

    @parameters begin
        tₛ = 0.5, [bounds = (1.e-3, Inf), tunable = true]
    end

    @constants begin
        m = 2.2, [tunable = false]    
        J = 0.05, [tunable = false]    
        r = 0.2, [tunable = false]    
        mg = 4., [tunable = false]    
        μ = 1., [tunable = false]    
    end
    
    eqs = [
        D(x₁(t)) ~ tₛ * (v₁(t))
        D(v₁(t)) ~ tₛ * (1 / m * (u₁(t) * cos(α(t)) - u₂(t) * sin(α(t))))
        D(x₂(t)) ~ tₛ * (v₂(t))
        D(v₂(t)) ~ tₛ * (1 / m * (-mg + u₁(t) * sin(α(t)) + u₂(t) * cos(α(t))))
        D(α(t)) ~ tₛ * (vα(t))
        D(vα(t)) ~ tₛ * (r / J * u₁(t))
    ]
    
    # scale the constraint grid
    constraint_grid = scaled_grids.constraint_grid[1:end - 1]
    
    grid_cons_u = [α(tᵢ) ≲ 30. for tᵢ in constraint_grid]
    grid_cons_l = [α(tᵢ) ≳ -30. for tᵢ in constraint_grid]
    
    cons = [
        x₁(last(tspan)) ~ 1.,
        v₁(last(tspan)) ~ 0.,
        x₂(last(tspan)) ~ 0.,
        v₂(last(tspan)) ~ 0.,
        α(last(tspan)) ~ 0.,
        vα(last(tspan)) ~ 0.,
        grid_cons_u...,
        grid_cons_l...
    ]
    
    costs = [
        Symbolics.Integral(t in (0.0, last(tspan)))(
            (2 * u₁(t)^2 + u₂(t)^2) + μ * tₛ
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
