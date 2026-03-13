function mountain_car(grids)

    num_states = 3
    num_controls = 1
    tspan = (0.,100.)

    scaled_grids = scale_grids!(tspan, grids)
    
    @variables begin
        x(..) = -0.5, [tunable = false]
        v(..) = 0., [tunable = false]
        obj(..) = 0., [tunable = false]
        u(..) = 0.5, [bounds = (-1., 1.), input = true]
    end
    
    @parameters begin
        tₛ = 1., [bounds = (1.e-3, Inf), tunable = true]
    end
    
    eqs = [
        D(x(t)) ~ tₛ * v(t)
        D(v(t)) ~ tₛ * (1.e-3 * u(t) - 2.5e-3 * cos(3 * x(t)))
        D(obj(t)) ~ tₛ
    ]
    
    cons = [
        x(last(tspan)) ~ 0.5,
        v(last(tspan)) ≳ 0.
    ]
    
    costs = [obj(last(tspan))]

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
	name = "Mountain Car"
    )

end
