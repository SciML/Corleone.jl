module cushioned_oscillation

using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using Symbolics
using ..OptimalControlBenchmarks: OptimalControlBenchmark

function make_problem()

    num_states = 3
    num_controls = 1
    
    @variables begin
        x(..) = 2.0, [tunable = false]
        v(..) = 5.0, [tunable = false]
        obj(..) = 0.0, [tunable = false]
        u(..) = 0.0, [bounds = (-5.0, 5.0), input = true]
    end
    @parameters begin
        tₛ = 1., [bounds = (1.e-3, Inf), tunable = true]
    end
    @constants begin
        m = 5., [tunable = false]
        c = 10., [tunable = false]
    end
    
    eqs = [
        D(x(t)) ~ tₛ * v(t)
        D(v(t)) ~ tₛ * (1 / m * (u(t) - c * x(t)))
        D(obj(t)) ~ tₛ
    ]
    
    # Define control discretization
    tspan = (0.,10.)
    dt = 0.2
    cgrid = collect(0.0:dt:last(tspan))[1:end-1]
    
    cons = [
        x(last(tspan)) ~ 0.,
        v(last(tspan)) ~ 0.,
    ]
    
    costs = [obj(1.)]

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
    :cushioned_oscillation,
    "Double integrator with quadratic control cost",
    make_problem
)

end