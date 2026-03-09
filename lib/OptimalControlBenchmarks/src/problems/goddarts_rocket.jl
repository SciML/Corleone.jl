module goddarts_rocket

using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using Symbolics
using ..OptimalControlBenchmarks: OptimalControlBenchmark

function make_problem()

    num_states = 3
    num_controls = 1
    
    @variables begin
        r(..) = 1., [tunable = false]
        v(..) = 0., [tunable = false]
        m(..) = 1., [tunable = false]
        u(..) = 0.5, [bounds = (0., 1.0), input = true]
    end
    
    @parameters begin
        T = 0.5, [bounds =(1.e-3, Inf), tunable = true]
    end
    
    @constants begin
        r_0 = 1., [tunable = false]
        v_0 = 0., [tunable = false]
        m_0 = 1., [tunable = false]
        r_T = 1.01, [tunable = false]
        b = 7., [tunable = false]
        T_max = 3.5, [tunable = false]
        A = 310., [tunable = false]
        k = 500., [tunable = false]
        C = 0.6, [tunable = false]
    end
    
    Drag = A * v(t)^2 * exp(-k * (r(t) - r_0))
    
    eqs = [
        D(r(t)) ~ T * v(t)
        D(v(t)) ~ T * (-1 / r(t)^2 + 1 / m(t) * (T_max * u(t) - Drag))
        D(m(t)) ~ T * (-b * u(t))
    ]
    
    # Define control discretization
    tspan = (0.,10.)
    dt = 0.1
    cgrid = collect(0.0:dt:last(tspan))
    
    grid_cons_u = [310. * v(tᵢ)^2 * 2.7182818^(-500. * (r(tᵢ) - 1.)) ≲ 0.6 for tᵢ in cgrid]
    
    cons = [
        r(last(tspan)) ~ 1.01,
        grid_cons_u...
    ]
    
    costs = [-m(last(tspan))]

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
    :goddarts_rocket,
    "Double integrator with quadratic control cost",
    make_problem
)

end