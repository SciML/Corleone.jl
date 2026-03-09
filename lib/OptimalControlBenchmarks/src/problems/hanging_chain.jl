module hanging_chain

using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using Symbolics
using ..OptimalControlBenchmarks: OptimalControlBenchmark

function make_problem()

    num_states = 3
    num_controls = 1

    @variables begin
        x₁(..) = 1., [tunable = false]
        x₂(..) = 0., [tunable = false]
        x₃(..) = 0., [tunable = false]
        u(..) = 0.5, [bounds = (-10., 20.), input = true]
    end

    @constants begin
        a = 1., [tunable = false]
        b = 3., [tunable = false]
        L_p = 4., [tunable = false]
    end

    eqs = [
        D(x₁(t)) ~ u(t)
        D(x₂(t)) ~ x₁(t) * sqrt(1 + u(t)^2)
        D(x₃(t)) ~ sqrt(1 + u(t)^2)
    ]

    # Define control discretization
    tspan = (0.,1.)
    dt = 0.02
    cgrid = collect(0.0:dt:last(tspan))[1:end-1]

    # grid_cons_u = [A * v(tᵢ)^2 * exp(-k * (r(tᵢ) - r_0)) ≲ 0.6 for tᵢ in cgrid]
    grid_cons_u1 = [x₁(tᵢ) ≳ 0. for tᵢ in cgrid]
    grid_cons_u2 = [x₂(tᵢ) ≳ 0. for tᵢ in cgrid]
    grid_cons_u3 = [x₃(tᵢ) ≳ 0. for tᵢ in cgrid]
    grid_cons_l1 = [x₁(tᵢ) ≲ 10. for tᵢ in cgrid]
    grid_cons_l2 = [x₂(tᵢ) ≲ 10. for tᵢ in cgrid]
    grid_cons_l3 = [x₃(tᵢ) ≲ 10. for tᵢ in cgrid]

    cons = [
        x₁(last(tspan)) - 3. ~ 0.,
        x₃(last(tspan)) - 4. ~ 0.,
        grid_cons_u1..., grid_cons_u2..., grid_cons_u3...,
        grid_cons_l1..., grid_cons_l2..., grid_cons_l3...
    ]

    costs = [x₂(last(tspan))]

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
    :hanging_chain,
    "Double integrator with quadratic control cost",
    make_problem
)

end