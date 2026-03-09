module van_der_pol

using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using Symbolics
using ..OptimalControlBenchmarks: OptimalControlBenchmark

function make_problem()

    num_states = 3
    num_controls = 1
    
    @variables begin
        x₁(..) = 0., [tunable = false]
        x₂(..) = 1., [tunable = false]
        u(..) = 0., [bounds = (-1., 1.), input = true]
    end
    
    eqs = [
        D(x₁(t)) ~ (1 - x₂(t)^2) * x₁(t) - x₂(t) + u(t)
        D(x₂(t)) ~ x₁(t)
    ]
    
    # Define control discretization
    tspan = (0.,10.)
    dt = 0.2
    cgrid = collect(0.0:dt:last(tspan))[1:end - 1]
    t_f = last(tspan)
    
    grid_cons_x₁ = [x₁(tᵢ) ≳ -0.25 for tᵢ in vcat(cgrid, t_f)]
    
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
        control_grid = cgrid,
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