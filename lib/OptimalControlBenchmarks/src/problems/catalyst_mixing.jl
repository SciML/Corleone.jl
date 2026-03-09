module catalyst_mixing

using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using Symbolics
using ..OptimalControlBenchmarks: OptimalControlBenchmark

function make_problem()

    num_states = 2
    num_controls = 1

        @variables begin
            x₁(..) = 1.0, [tunable = false]
            x₂(..) = 0.0, [tunable = false]
            w(..) = 0.0, [bounds = (0.0, 1.0), input = true]
        end
        
        eqs = [
            D(x₁(t)) ~ w(t) * (10 * x₂(t) - x₁(t))
            D(x₂(t)) ~ w(t) * (x₁(t) - 10 * x₂(t)) - (1 - w(t)) * x₂(t)
        ]

    # Define control discretization
    tspan = (0.,1.)
    dt = 0.02
    cgrid = collect(0.0:dt:last(tspan))[1:end-1]

    costs = [-1 + x₁(last(tspan)) + x₂(last(tspan))]

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
    :catalyst_mixing,
    "Double integrator with quadratic control cost",
    make_problem
)

end