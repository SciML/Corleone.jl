module robbins

using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using Symbolics
using ..OptimalControlBenchmarks: OptimalControlBenchmark

function make_problem(constraint_grid=collect(0.:0.1:1.))

    num_states = 4
    num_controls = 1
    tspan = (0.,10.)

    @variables begin
        x₁(..) = 1., [tunable = false]
        x₂(..) = -2., [tunable = false]
        x₃(..) = 0., [tunable = false]
        u(..) = 0., [input = true]
    end

    @constants begin
        α = 3., [tunable = false]
        β = 0., [tunable = false]
        γ = 0.5, [tunable = false]
    end

    eqs = [
        D(x₁(t)) ~ x₂(t)
        D(x₂(t)) ~ x₃(t)
        D(x₃(t)) ~ u(t)
    ]

    # scale the constraint grid
    constraint_grid = constraint_grid * (last(tspan) - first(tspan))
    constraint_grid = (constraint_grid .+ first(tspan))[1:end - 1]

    grid_cons_x₁ = [x₃(tᵢ) ≳ 0. for tᵢ in constraint_grid]

    cons = [
        x₁(last(tspan)) ~ 0.,
        x₂(last(tspan)) ~ 0.,
        x₃(last(tspan)) ~ 0.,
        grid_cons_x₁...
    ]

    costs = [
        Symbolics.Integral(t in (0.0, last(tspan)))(
            α * x₁(t) + β * x₁(t)^2 + γ * u(t)^2
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
    :robbins,
    "Double integrator with quadratic control cost",
    make_problem
)

end