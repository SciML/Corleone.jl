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
    optimized_stages, starting_control, starting_objective, optimized_objective,
    lower_bounds, upper_bounds, grid, ncontrols, initial_state, optimum, report

@testset "Manual single-shooting reproduction" begin
    @test successful_retcode(optimum)
    @test length(optimized_stages) == ncontrols == 24
    @test all(successful_retcode, optimized_stages)
    @test all(sol -> all(u -> all(isfinite, u), sol.u), optimized_stages)
    @test all(isfinite, optimized_control)
    @test all(lower_bounds .- 1.0e-8 .<= optimized_control .<= upper_bounds .+ 1.0e-8)
    @test isfinite(starting_objective) && isfinite(optimized_objective)
    @test optimized_objective < starting_objective - 1.0e-3
    @test fishing_objective(starting_control, nothing) ≈ starting_objective
    @test optimized_stages[1].prob.u0 == initial_state
    for (i, sol) in enumerate(optimized_stages)
        @test sol.prob.tspan == (grid[i], grid[i + 1])
        @test sol.prob.p == optimized_control[i]
        @test sol.t == [grid[i], grid[i + 1]]
        i == 1 || @test sol.prob.u0 == optimized_stages[i - 1].u[end]
    end
    # Independent central directional difference verifies AD through stage
    # construction and propagation, without requiring another AD backend.
    direction = collect(range(-0.5, 0.5; length = ncontrols))
    gradient = ForwardDiff.gradient(c -> fishing_objective(c, nothing), starting_control)
    delta = 1.0e-5
    difference = (
        fishing_objective(starting_control .+ delta .* direction, nothing) -
        fishing_objective(starting_control .- delta .* direction, nothing)
    ) / (2delta)
    @test all(isfinite, gradient)
    @test isapprox(sum(gradient .* direction), difference; atol = 1.0e-5, rtol = 1.0e-4)
end
@info "Independent tutorial reproduction" report
