module moon_landing

using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using Symbolics
using ..OptimalControlBenchmarks: OptimalControlBenchmark

function make_problem()

    num_states = 3
    num_controls = 1
    
    @variables begin
        h(..) = 1., [tunable = false]
        v(..) = -0.783, [tunable = false]
        m(..) = 1., [tunable = false]
        T(..) = 0.5, [bounds = (0., 1.227), input = true]
    end
    
    @parameters begin
        tₛ = 1., [bounds = (1.e-3, Inf), tunable = true]
    end
    
    eqs = [
        D(h(t)) ~ tₛ * v(t)
        D(v(t)) ~ tₛ * (-1 + T(t) / m(t))
        D(m(t)) ~ tₛ * (- T(t) / 2.349)
    ]
    
    # Define control discretization
    tspan = (0.,1.)
    dt = 0.02
    cgrid = collect(0.0:dt:last(tspan))
    
    cons = [
        h(last(tspan)) ~ 0.,
        v(last(tspan)) ~ 0.
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
        control_grid = cgrid,
        num_states = num_states,
        num_controls = num_controls
    )

end


benchmark = OptimalControlBenchmark(
    :moon_landing,
    "Double integrator with quadratic control cost",
    make_problem
)

end