function goddarts_rocket(grids)

    num_states = 3
    num_controls = 1
    tspan = (0.,10.)

    scaled_grids = scale_grids!(tspan, grids)
    
    @variables begin
        r(..) = 1., [tunable = false]
        v(..) = 0., [tunable = false]
        m(..) = 1., [tunable = false]
        u(..) = 0.5, [bounds = (0., 1.0), input = true]
    end
    
    @parameters begin
        T = 0.5, [bounds =(1.e-3, Inf), tunable = true]
    end
    
    @constants begin
        r_0 = 1., [tunable = false]
        v_0 = 0., [tunable = false]
        m_0 = 1., [tunable = false]
        r_T = 1.01, [tunable = false]
        b = 7., [tunable = false]
        T_max = 3.5, [tunable = false]
        A = 310., [tunable = false]
        k = 500., [tunable = false]
        C = 0.6, [tunable = false]
    end
    
    Drag = A * v(t)^2 * exp(-k * (r(t) - r_0))
    
    eqs = [
        D(r(t)) ~ T * v(t)
        D(v(t)) ~ T * (-1 / r(t)^2 + 1 / m(t) * (T_max * u(t) - Drag))
        D(m(t)) ~ T * (-b * u(t))
    ]
    
    # scale the constraint grid
    constraint_grid = scaled_grids.constraint_grid
    
    grid_cons_u = [310. * v(tᵢ)^2 * 2.7182818^(-500. * (r(tᵢ) - 1.)) ≲ 0.6 for tᵢ in constraint_grid]
    
    cons = [
        r(last(tspan)) ~ 1.01,
        grid_cons_u...
    ]
    
    costs = [-m(last(tspan))]

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
