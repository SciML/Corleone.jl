module tubular_reactor

using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using Symbolics
using ..OptimalControlBenchmarks: OptimalControlBenchmark

function make_problem()

    num_states = 2
    num_controls = 1
    
    @variables begin
        x₁(..) = 1., [tunable = false]
        x₂(..) = 0., [tunable = false]
        u(..) = 0., [bounds = (0., 5.), input = true]
    end
    
    eqs = [
        D(x₁(t)) ~ -(u(t) + 0.5 * u(t)^2) * x₁(t)
        D(x₂(t)) ~ u(t) * x₁(t)
    ]
    
    # Define control discretization
    tspan = (0.,1.)
    dt = 0.02
    cgrid = collect(0.0:dt:last(tspan))[1:end - 1]

    costs = [-x₂(last(tspan))]

    @named oc_problem = System(
        eqs,
        t;
        costs = costs
    )

    return (
        system = oc_problem,
        control_grid = cgrid,
        num_states = num_states,
        num_controls = num_controls
    )

end


benchmark = OptimalControlBenchmark(
    :tubular_reactor,
    "Double integrator with quadratic control cost",
    make_problem
)

end