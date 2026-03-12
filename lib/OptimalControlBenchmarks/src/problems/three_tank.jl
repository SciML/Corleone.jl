module three_tank

using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using Symbolics
using ..OptimalControlBenchmarks: OptimalControlBenchmark

function make_problem(constraint_grid=collect(0.:0.1:1.))

    num_states = 4
    num_controls = 3
    tspan = (0.,12.)
    
    @variables begin
        x₁(..) = 2., [tunable = false]
        x₂(..) = 2., [tunable = false]
        x₃(..) = 2., [tunable = false]
        w₁(..) = 1. / 3., [bounds = (0., 1.), input = true]
        w₂(..) = 1. / 3., [bounds = (0., 1.), input = true]
        w₃(..) = 1. / 3., [bounds = (0., 1.), input = true]
    end
    
    @constants begin
        c₁ = 1., [tunable = false]
        c₂ = 2., [tunable = false]
        c₃ = 0.8, [tunable = false]
        k₁ = 2., [tunable = false]
        k₂ = 3., [tunable = false]
        k₃ = 1., [tunable = false]
        k₄ = 3., [tunable = false]
    end
    
    eqs = [
        D(x₁(t)) ~ -sqrt(x₂(t)) + c₁ * w₁(t) + c₂ * w₂(t) - w₃(t) * sqrt(c₃ * x₁(t))
        D(x₂(t)) ~ sqrt(x₁(t)) - sqrt(x₂(t))
        D(x₃(t)) ~ sqrt(x₂(t)) - sqrt(x₃(t)) + w₃(t) * sqrt(c₃ * x₁(t))
    ]
    
    # scale the constraint grid
    constraint_grid = constraint_grid * (last(tspan) - first(tspan))
    constraint_grid = (constraint_grid .+ first(tspan))

    grid_cons_w = [w₁(tᵢ) + w₂(tᵢ) + w₃(tᵢ) ~ 1. for tᵢ in constraint_grid]
    
    cons = [
        grid_cons_w...
    ]
    
    costs = [
        Symbolics.Integral(t in (0.0, last(tspan)))(
            k₁ * (x₂(t) - k₂)^2 + k₃ * (x₃(t) - k₄)^2
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
    :three_tank,
    "Double integrator with quadratic control cost",
    make_problem
)

end