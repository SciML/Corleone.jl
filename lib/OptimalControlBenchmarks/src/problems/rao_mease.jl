module rao_mease

using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using Symbolics
using ..OptimalControlBenchmarks: OptimalControlBenchmark

function make_problem(constraint_grid=nothing)

    num_states = 2
    num_controls = 1
    tspan = (0.,10)

    @variables begin
        x(..) = 1., [tunable = false]
        u(..) = 0., [input = true]
    end

    eqs = [
        D(x(t)) ~ -x(t)^3 + u(t)
    ]

    cons = [
        x(last(tspan)) ~ 1.5
    ]

    costs = [
        Symbolics.Integral(t in (0.0, last(tspan)))(
            x(t)^2 + u(t)^2
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
        tspan = tspan,
        num_states = num_states,
        num_controls = num_controls
    )

end


benchmark = OptimalControlBenchmark(
    :rao_mease,
    "Double integrator with quadratic control cost",
    make_problem
)

end