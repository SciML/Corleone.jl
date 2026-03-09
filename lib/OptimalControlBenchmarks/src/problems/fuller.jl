module fuller

using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using Symbolics
using ..OptimalControlBenchmarks: OptimalControlBenchmark

function make_problem()

    num_states = 3
    num_controls = 1

    @variables begin
        x₀(..) = 0.01, [tunable = false]
        x₁(..) = 0.0, [tunable = false]
        u(..) = 0.5, [bounds = (0., 1.0), input = true]
    end

    eqs = [
        D(x₀(t)) ~ x₁(t)
        D(x₁(t)) ~ 1. - 2. * u(t)
    ]

    # Define control discretization
    tspan = (0.,1.)
    dt = 0.02
    cgrid = collect(0.0:dt:last(tspan))

    cons = [
        x₀(last(tspan)) ~ 0.01,
        x₁(last(tspan)) ~ 0.,
    ]

    costs = [
        Symbolics.Integral(t in (0.0, last(tspan)))(
            x₀(t)^2
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
        control_grid = cgrid,
        num_states = num_states,
        num_controls = num_controls
    )

end


benchmark = OptimalControlBenchmark(
    :fuller,
    "Double integrator with quadratic control cost",
    make_problem
)

end