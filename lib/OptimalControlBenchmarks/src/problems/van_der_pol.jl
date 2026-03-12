module van_der_pol

using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using Symbolics
using ..OptimalControlBenchmarks: OptimalControlBenchmark

function make_problem(constraint_grid=collect(0.:0.1:1.))

    num_states = 3
    num_controls = 1
    tspan = (0.,10.)
    
    @variables begin
        x₁(..) = 0., [tunable = false]
        x₂(..) = 1., [tunable = false]
        u(..) = 0., [bounds = (-1., 1.), input = true]
    end
    
    eqs = [
        D(x₁(t)) ~ (1 - x₂(t)^2) * x₁(t) - x₂(t) + u(t)
        D(x₂(t)) ~ x₁(t)
    ]
    
    # scale the constraint grid
    constraint_grid = constraint_grid * (last(tspan) - first(tspan))
    constraint_grid = (constraint_grid .+ first(tspan))
    
    grid_cons_x₁ = [x₁(tᵢ) ≳ -0.25 for tᵢ in constraint_grid]
    
    cons = [
        grid_cons_x₁...
    ]
    
    costs = [
        Symbolics.Integral(t in (0.0, last(tspan)))(
            x₁(t)^2 + x₂(t)^2 + u(t)^2
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
    :van_der_pol,
    "Double integrator with quadratic control cost",
    make_problem
)

end