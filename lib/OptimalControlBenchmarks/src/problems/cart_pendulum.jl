module cart_pendulum

using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using Symbolics
using ..OptimalControlBenchmarks: OptimalControlBenchmark

function make_problem()

    num_states = 4
    num_controls = 1

    @variables begin
        x(..) = 0.0, [tunable = false]
        θ(..) = 0.0, [tunable = false]
        dx(..) = 0.0, [tunable = false]
        dtheta(..) = 0.0, [tunable = false]
        w(..) = 0.0, [bounds=(-30,30), input = true]
    end
    
    @parameters begin
        α = 10., [tunable = false]
        β = 50., [tunable = false]
        γ = 0.5, [tunable = false]
        M = 1., [tunable = false]
        m = 0.1, [tunable = false]
        g = 9.81, [tunable = false]
    end

    eqs = [
        D(x(t)) ~ dx(t)
        D(θ(t)) ~ dtheta(t)
        D(dx(t)) ~ (w(t) + m * g * sin(θ(t)) * cos(θ(t)) + m * dtheta(t)^2 * sin(θ(t))) / (M + m * (1 - cos(θ(t)))^2)
        D(dtheta(t)) ~ -g * sin(θ(t)) - ((w(t) + m * g * sin(θ(t)) * cos(θ(t)) + m * dtheta(t) * sin(θ(t))) / (M + m * (1 - cos(θ(t))^2))) * cos(θ(t))
    ]
    
    # Define control discretization
    tspan = (0.,4.)
    dt = (last(tspan) - first(tspan)) / 40
    cgrid = collect(0.0:dt:last(tspan))[1:end-1]
    
    grid_cons_le = [x(tᵢ) ≲ 2. for tᵢ in vcat(cgrid, last(tspan))]
    grid_cons_ge = [x(tᵢ) ≳ -2. for tᵢ in vcat(cgrid, last(tspan))]

    cons = [
        grid_cons_le...,
        grid_cons_ge...
    ]

    costs = [
        Symbolics.Integral(t in (0.0, 1.0))(
            α * x(t)^2 + β * (θ(t) - pi)^2 + γ * w(t)^2
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
    :cart_pendulum,
    "Double integrator with quadratic control cost",
    make_problem
)

end