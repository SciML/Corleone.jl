module double_oscillator

using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using Symbolics
using ..OptimalControlBenchmarks: OptimalControlBenchmark

function make_problem()

    num_states = 5
    num_controls = 1

    @variables begin
        x₀(..) = 0.0, [tunable = false]
        x₁(..) = 0.0, [tunable = false]
        x₂(..) = 0.0, [tunable = true]
        x₃(..) = 0.0, [tunable = true]
        u(..) = 1.0, [bounds = (-1.0, 1.0), input = true]
    end

    @constants begin
        m₁ = 200., [tunable = false]
        m₂ = 2., [tunable = false]
        k₁ = 100., [tunable = false]
        k₂ = 3., [tunable = false]
        c = 0.5, [tunable = false]
        T = 2 * pi, [tunable = false]
    end

    eqs = [
        D(x₀(t)) ~ x₂(t)
        D(x₁(t)) ~ x₃(t)
        D(x₂(t)) ~ - (k₁ + k₂) / m₁ * x₀(t) + k₂ / m₁ * x₁(t) + 1 / m₁ * sin(2 * pi / T * t)
        D(x₃(t)) ~ k₂ / m₂ * x₀(t) - k₂ / m₂ * x₁(t) - c * (1 - u(t)) / m₂ * x₃(t)
    ]

    # Define control discretization
    tspan = (0.,2 * pi)
    dt = 0.1
    cgrid = collect(0.0:dt:last(tspan))[1:end-1]

    costs = [
        Symbolics.Integral(t in (0.0, 1.0))(
            0.5 * (x₀(t)^2 + x₁(t)^2 + u(t)^2)
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
    :double_oscillator,
    "Double integrator with quadratic control cost",
    make_problem
)

end