module ocean

using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using Symbolics
using ..OptimalControlBenchmarks: OptimalControlBenchmark

function make_problem(constraint_grid=collect(0.:0.1:1.))

    num_states = 3
    num_controls = 2
    tspan = (0.,400.)

    @variables begin
        S(..) = 2.e3, [tunable = false]
        R(..) = 1.e4, [tunable = false]
        u₁(..) = 7., [bounds = (0., 40.), input = true]
        u₂(..) = 7., [bounds = (0., 40.), input = true]
    end

    @constants begin
        ρ = 3.e-2, [tunable = false]
        γ = 1.e-3, [tunable = false]
        ω = 0.1, [tunable = false]
        b = 50., [tunable = false]
        μ = 0.5, [tunable = false]
        a₁ = 2., [tunable = false]
        a₂ = 2., [tunable = false]
        ν = 1., [tunable = false]
        c₁ = 50., [tunable = false]
        c₂ = 4.e-3, [tunable = false]
        Sₚ = 6.e2, [tunable = false]
        S₀ = 2.e3, [tunable = false]
        R₀ = 1.e4, [tunable = false]
        Dₗ₀ = 2.3e4, [tunable = false]
    end

    # auxiliary functions
    U = b * u₁(t) - μ * u₁(t)^2
    A = a₁ * u₂(t) + a₂ * u₂(t)^2
    C = c₁ - c₂ * R(t)
    Df = ν * (0.3 * S(t) - Sₚ)^2
    Dₗ = Dₗ₀ + R₀ + S₀ - R(t) - S(t)

    eqs = [
        D(R(t)) ~ u₁(t) - u₂(t) - γ * (S(t) - ω * Dₗ)
        D(S(t)) ~ -u₁(t)
    ]

    # Define control discretization
    constraint_grid = constraint_grid * (last(tspan) - first(tspan))
    constraint_grid = (constraint_grid .+ first(tspan))

    grid_cons_Su = [S(tᵢ) ≲ 1.e5 for tᵢ in constraint_grid]
    grid_cons_Ru = [R(tᵢ) ≲ 1.e5 for tᵢ in constraint_grid]
    grid_cons_Sl = [S(tᵢ) ≳ 0. for tᵢ in constraint_grid]
    grid_cons_Rl = [R(tᵢ) ≳ 0. for tᵢ in constraint_grid]

    cons = [
        grid_cons_Su..., grid_cons_Ru...,
        grid_cons_Sl..., grid_cons_Rl...
    ]

    costs = [
        Symbolics.Integral(t in (0.0, last(tspan)))(
            -exp(-ρ * t) * (U - A - u₁(t) * C - Df)
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
    :ocean,
    "Double integrator with quadratic control cost",
    make_problem
)

end