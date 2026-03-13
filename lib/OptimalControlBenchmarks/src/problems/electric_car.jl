function electric_car(grids)

    num_states = 4
    num_controls = 1
    tspan = (0.,10.)

    scaled_grids = scale_grids!(tspan, grids)
    
    @variables begin
        x_1(..) = 0.0, [tunable = false]
        x_2(..) = 0.0, [tunable = false]
        x_3(..) = 0.0, [tunable = false]
        u(..) = 0.0, [bounds = (-5.0, 5.0), input = true]
    end
    
    @constants begin
        K_r = 10., [tunable = false]    
        rho = 1.293, [tunable = false]    
        C_x = 0.4, [tunable = false]    
        S = 2., [tunable = false]    
        r = 0.33, [tunable = false]    
        K_f = 0.03, [tunable = false]    
        K_m = 0.27, [tunable = false]    
        R_m = 0.03, [tunable = false]    
        L_m = 0.05, [tunable = false]    
        M = 250., [tunable = false]    
        g = 9.81, [tunable = false]    
        V_alim = 150., [tunable = false]    
        R_bat = 0.05, [tunable = false]    
    end
    
    temp = K_m * x_1(t)
    temp += - r / K_r * (M * g * K_f + 0.5 * rho * S * C_x * r^2 / K_r^2 *  x_2(t)^2)
    temp *= K_r^2 / (M * r^2)
    
    eqs = [
        D(x_1(t)) ~ (V_alim * u(t) - R_m * x_1(t) - K_m * x_2(t)) / L_m 
        D(x_2(t)) ~ temp
        D(x_3(t)) ~ r / K_r * x_2(t)
    ]
    
    constraint_grid = scaled_grids.constraint_grid
    
    grid_cons_u = [x_1(tᵢ) ≲ 150. for tᵢ in constraint_grid]
    grid_cons_l = [x_1(tᵢ) ≳ -150. for tᵢ in constraint_grid]
    
    cons = [
        x_3(last(tspan)) ~ 100.,
        grid_cons_u...,
        grid_cons_l...
    ]
    
    costs = [
        Symbolics.Integral(t in (0.0, last(tspan)))(
            V_alim * u(t) * x_1(t) + R_bat * x_1(t)^2
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
	name = "Electric Car"
    )

end
