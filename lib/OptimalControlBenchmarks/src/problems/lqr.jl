module lqr

using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using Symbolics
using ..OptimalControlBenchmarks: OptimalControlBenchmark

function make_problem(constraint_grid=nothing)

    num_states = 2
    num_controls = 1
    tspan = (0.,10.)
    
    @variables begin
        x(..) = 1., [tunable = false]
        u(..) = 0.5, [input = true]
    end
    
    @constants begin
        a = -1., [tunable = false]
        b = 1., [tunable = false]
    end
    
    eqs = [
        D(x(t)) ~ a * x(t) + b * u(t)
    ]
        
    costs = [
        Symbolics.Integral(t in (0.0, last(tspan)))(
            10. * (x(t) - 3.)^2 + 0.1 * u(t)^2
        ),
    ]

    @named oc_problem = System(
        eqs,
        t;
        costs = costs
    )

    return (
        system = oc_problem,
        tspan = tspan,
        num_states = num_states,
        num_controls = num_controls
    )

end


benchmark = OptimalControlBenchmark(
    :lqr,
    "Double integrator with quadratic control cost",
    make_problem
)

end