module bryson_denham

using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using Symbolics
using ..OptimalControlBenchmarks: OptimalControlBenchmark

function make_problem()

    num_states = 2
    num_controls = 1

    @variables begin
        x(..) = 0.0, [tunable = false]
        v(..) = 1.0, [tunable = false]
        w(..) = 0.0, [input = true]
    end

    eqs = [
        D(x(t)) ~ v(t)
        D(v(t)) ~ w(t)
    ]

    tspan = (0.,1.)
    dt = 0.01
    cgrid = collect(0.0:dt:last(tspan))[1:end-1]

    grid_cons = [x(tᵢ) ≲ 1/9 for tᵢ in cgrid]

    cons = [
        x(last(tspan)) ~ 0.,
        v(last(tspan)) ~ -1.,
        grid_cons...
    ]

    costs = [
        Symbolics.Integral(t in (0.0, 1.0))(w(t)^2)
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
    :bryson_denham,
    "Double integrator with quadratic control cost",
    make_problem
)

end