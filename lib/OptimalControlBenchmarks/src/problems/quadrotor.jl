module quadrotor

using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using Symbolics
using ..OptimalControlBenchmarks: OptimalControlBenchmark

function make_problem()

    num_states = 7
    num_controls = 4
    
    @variables begin
        x₁(..) = 0., [tunable = false]
        x₂(..) = 0., [tunable = false]
        x₃(..) = 1., [tunable = false]
        x₄(..) = 0., [tunable = false]
        x₅(..) = 0., [tunable = false]
        x₆(..) = 0., [tunable = false]
        w₁(..) = 1. / 3., [bounds = (0., 1.), input = true]
        w₂(..) = 1. / 3., [bounds = (0., 1.), input = true]
        w₃(..) = 1. / 3., [bounds = (0., 1.), input = true]
        u(..) = 0., [bounds = (0., 1.e-3), input = true]
    end
    
    @constants begin
        g = 9.81, [tunable = false]
        M = 1.3, [tunable = false]
        L = 0.305, [tunable = false]
        I = 0.0605, [tunable = false]
    end
    
    eqs = [
        D(x₁(t)) ~ x₂(t)
        D(x₂(t)) ~ g * sin(x₅(t)) + w₁(t) * u(t) * sin(x₅(t)) / M
        D(x₃(t)) ~ x₄(t)
        D(x₄(t)) ~ g * cos(x₅(t)) - g + w₁(t) * u(t) * cos(x₅(t)) / M
        D(x₅(t)) ~ x₆(t)
        D(x₆(t)) ~ -w₂(t) * L * u(t) / I + w₃(t) * L * u(t) / I
    ]
    
    # Define control discretization
    tspan = (0.,7.5)
    dt = 0.15
    cgrid = collect(0.0:dt:last(tspan))[1:end - 1]
    t_f = last(tspan)
    
    grid_cons_w = [w₁(tᵢ) + w₂(tᵢ) + w₃(tᵢ) ~ 1. for tᵢ in vcat(cgrid, t_f)]
    grid_cons_x₃ = [x₃(tᵢ) ≳ 0. for tᵢ in vcat(cgrid, t_f)]
    
    cons = [
        grid_cons_w...,
        grid_cons_x₃...
    ]
    
    costs = [
        5 * (x₁(t_f) - 6)^2 + 5 * (x₃(t_f) - 1)^2 + 5 * (sin(x₅(t_f) * 0.5))^2 + Symbolics.Integral(t in (0.0, last(tspan)))(
            5 * u(t)^2
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
    :quadrotor,
    "Double integrator with quadratic control cost",
    make_problem
)

end