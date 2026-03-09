module goddarts_rocket

using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using Symbolics
using ..OptimalControlBenchmarks: OptimalControlBenchmark

function make_problem()

    num_states = 3
    num_controls = 1

    @variables begin
        x₀(..) = 0.5, [tunable = false]
        x₁(..) = 0.7, [tunable = false]
        u(..) = 0.5, [bounds = (0., 1.), input = true]
    end

    @constants begin
        c₀ = 0.4, [tunable = false]
        c₁ = 0.2, [tunable = false]
    end

    eqs = [
        D(x₀(t)) ~ x₀(t) - x₀(t) * x₁(t) - c₀ * x₀(t) * u(t)
        D(x₁(t)) ~ -x₁(t) + x₀(t) * x₁(t) - c₁ * x₁(t) * u(t)
    ]

    # Define control discretization
    tspan = (0.,12.)
    dt = 0.24
    cgrid = collect(0.0:dt:last(tspan))

    costs = [
        Symbolics.Integral(t in (0.0, last(tspan)))(
            (x₀(t) - 1.)^2 + (x₁(t) - 1.)^2
        ),
    ]

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
    :goddarts_rocket,
    "Double integrator with quadratic control cost",
    make_problem
)

end