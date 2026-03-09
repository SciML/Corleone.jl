module egerstedt

using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using Symbolics
using ..OptimalControlBenchmarks: OptimalControlBenchmark

function make_problem()

    num_states = 3
    num_controls = 3

    @variables begin
        x(..) = 0.5, [tunable = false]
        y(..) = 0.5, [tunable = false]
        u_1(..) = 1 / 3, [bounds = (-1.0, 1.0), input = true]
        u_2(..) = 1 / 3, [bounds = (-1.0, 1.0), input = true]
        u_3(..) = 1 / 3, [bounds = (-1.0, 1.0), input = true]
    end

    eqs = [
        D(x(t)) ~ -x(t) * u_1(t) + (x(t) + y(t)) * u_2(t) + (x(t) - y(t)) * u_3(t)
        D(y(t)) ~ (x(t) + 2 * y(t)) * u_1(t) + (x(t) - 2 * y(t)) * u_2(t) + (x(t) + y(t)) * u_3(t)
    ]

    # Define control discretization
    tspan = (0.,1.)
    dt = 0.1
    cgrid = collect(0.0:dt:last(tspan))[1:end-1]

    grid_cons = [u_1(tᵢ) + u_2(tᵢ) + u_3(tᵢ) ~ 1. for tᵢ in cgrid]

    cons = [
        grid_cons...
    ]

    costs = [
        Symbolics.Integral(t in (0.0, 1.0))(
            x(t)^2 + y(t)^2
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
    :egerstedt,
    "Double integrator with quadratic control cost",
    make_problem
)

end