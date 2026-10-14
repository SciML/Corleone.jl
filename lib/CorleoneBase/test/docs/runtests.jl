using Test
using ForwardDiff
using SciMLBase: successful_retcode

# Documenter's qualified @docs bindings are evaluated in Main. Match the shared
# manual's entry point even when SafeTestsets puts this test body in a module.
Base.include(Main, joinpath(@__DIR__, "..", "..", "..", "..", "docs", "check_corleonebase.jl"))

# Execute the source again independently of Literate's evaluation module.
# This also checks repeated objective calls do not retain accumulated state.
module FishingReproduction
    include(joinpath(@__DIR__, "..", "..", "examples", "lotka_fishing", "main.jl"))
end
using .FishingReproduction: fishing_stages, fishing_objective, optimized_control,
    optimized_stages, controls, starting_objective, optimized_objective,
    lower_bounds, upper_bounds, grid, tspans, initial_state, optimum, fishing_sequence

@testset "Manual single-shooting reproduction" begin
    @test successful_retcode(optimum)
    @test fishing_stages isa Function && fishing_objective isa Function
    @test grid == collect(0.0:0.1:12.0)
    @test length(optimized_stages) == length(controls) == 120
    @test all(successful_retcode, optimized_stages)
    @test all(sol -> all(u -> all(isfinite, u), sol.u), optimized_stages)
    @test all(isfinite, optimized_control)
    @test all(lower_bounds .- 1.0e-8 .<= optimized_control .<= upper_bounds .+ 1.0e-8)
    @test isfinite(starting_objective) && isfinite(optimized_objective)
    @test optimized_objective < starting_objective - 1.0e-3
    @test fishing_objective(controls, fishing_sequence.states) ≈ starting_objective
    @test optimized_stages[1].prob.u0 == initial_state
    for (i, sol) in enumerate(optimized_stages)
        @test sol.prob.tspan == (grid[i], grid[i + 1])
        @test sol.prob.p == optimized_control[i]
        @test sol.t[end] == grid[i + 1]
        @test i == 1 ? sol.t[1] == grid[i] : sol.t[1] > grid[i]
        i == 1 || @test sol.prob.u0 == optimized_stages[i - 1].u[end]
    end
    # Independent central directional difference verifies AD through stage
    # construction and propagation, without requiring another AD backend.
    # Differentiate one scalar perturbation of ALL 120 controls. This still
    # exercises dual-valued stage construction and propagation over the full
    # horizon, but uses one solve sweep rather than a 120-variable gradient.
    direction = collect(range(-0.5, 0.5; length = length(controls)))
    along_direction = alpha -> fishing_objective(controls .+ alpha .* direction, fishing_sequence.states)
    derivative = ForwardDiff.derivative(along_direction, 0.0)
    delta = 1.0e-5
    difference = (
        along_direction(delta) - along_direction(-delta)
    ) / (2delta)
    @test isfinite(derivative)
    @test isapprox(derivative, difference; atol = 1.0e-5, rtol = 1.0e-4)
end
@info "Independent tutorial reproduction" starting_objective optimized_objective
