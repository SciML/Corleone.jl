module f8

using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using Symbolics
using ..OptimalControlBenchmarks: OptimalControlBenchmark

function make_problem(constraint_grid=nothing)

    num_states = 4
    num_controls = 1
    tspan = (0.,10.)

    @variables begin
        x_0(..) = 0.4655, [tunable = false]
        x_1(..) = 0.0, [tunable = false]
        x_2(..) = 0.0, [tunable = false]
        obj(..) = 0.0, [tunable = false]
        u(..) = 0.5, [bounds = (0., 1.0), input = true]
    end

    @parameters begin
        T = 0.5, [bounds =(1.e-3, Inf), tunable = true]
    end

    @constants begin
        ξ = 0.05236, [tunable = false]
    end

    du_1 = -0.877 * x_0(t) + x_2(t) - 0.088 * x_0(t) * x_2(t) + 0.47 * x_0(t)^2 - 0.019 * x_1(t)^2
    du_1 += - x_0(t)^2 * x_2(t) + 3.846 * x_0(t)^3
    du_1 += 0.215 * ξ - 0.28 * x_0(t)^2 * ξ + 0.47 * x_0(t) * ξ^2 - 0.63 * ξ^3
    du_1 += - (0.215 * ξ - 0.28 * x_0(t)^2 * ξ - 0.63 * ξ^3) * 2 * u(t)
    du_3 = -4.208 * x_0(t) - 0.396 * x_2(t) - 0.47 * x_0(t)^2 - 3.564 * x_0(t)^3
    du_3 += 20.967 * ξ - 6.265 * x_0(t)^2 * ξ + 46. * x_0(t) * ξ^2 - 61.4 * ξ^3
    du_3 += -(20.967 * ξ - 6.265 * x_0(t)^2 * ξ - 61.4 * ξ^3) * 2 * u(t)


    eqs = [
        D(x_0(t)) ~ T * du_1
        D(x_1(t)) ~ T * x_2(t)
        D(x_2(t)) ~ T * du_3
        D(obj(t)) ~ T
    ]

    cons = [
        x_0(last(tspan)) ~ 0.,
        x_1(last(tspan)) ~ 0.,
        x_2(last(tspan)) ~ 0.,
    ]

    costs = [obj(last(tspan))]

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
    :f8,
    "Double integrator with quadratic control cost",
    make_problem
)

end