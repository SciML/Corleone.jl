module denbigh

using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using Symbolics
using ..OptimalControlBenchmarks: OptimalControlBenchmark

function make_problem()

    # We start by defining our system
    num_states = 3
    num_controls = 1

    @variables begin
        x₁(..) = 1.0, [tunable = false]
        x₂(..) = 0.0, [tunable = false]
        x₃(..) = 0.0, [tunable = false]
        T(..) = 300.0, [bounds = (273.0, 415.0), input = true]
    end
    @parameters begin
        tₛ = 1., [bounds = (1.e-3, Inf), tunable = true]
    end
    @constants begin
        E[1:4] = [3.e3, 6.e3, 3.e3, 0.], [tunable = false]
        k⁰[1:4] = [1.e3, 1.e7, 1.e1, 1.e-3], [tunable = false]
    end

    # auxiliary equations for the kᵢ
    k = [k⁰[i] * exp(-E[i] / T(t)) for i in [1,2,3,4]]

    eqs = [
        D(x₁(t)) ~ -k[1] * x₁(t) - k[2] * x₁(t)
        D(x₂(t)) ~ k[1] * x₁(t) - k[3] + k[4] * x₂(t)
        D(x₃(t)) ~ k[3] * x₂(t)
    ]

    # Define control discretization
    tspan = (0.,1000.)
    dt = 10.
    cgrid = collect(0.0:dt:last(tspan))[1:end-1]

    costs = [-x₃(last(tspan))]

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
    :denbigh,
    "Double integrator with quadratic control cost",
    make_problem
)

end