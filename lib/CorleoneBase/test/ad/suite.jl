module SequentialADTests

using CorleoneBase
import CommonSolve
import DifferentiationInterface as DI
using OrdinaryDiffEqTsit5: Tsit5
using SciMLBase: ODEProblem, isinplace, remake, successful_retcode
using Test

export test_sequential_ad

include("fixtures.jl")

function test_primal(case, x; solve_kwargs)
    solutions = sequential_solutions(case, x; solve_kwargs)
    y, _ = reference_endpoints(case, x)
    @test length(solutions) == 3
    length(solutions) == 3 || return nothing
    for i in 1:3
        sol = solutions[i]
        @test successful_retcode(sol)
        @test isinplace(sol.prob) == is_inplace(case)
        @test sol.prob.tspan == (Float64(i - 1), Float64(i))
        # Also checks that saving options reach all three inner solves.
        @test sol.t == [Float64(i - 1), Float64(i)]
        @test sol.prob.p == [x[i == 1 ? 2 : 3]]
        expected_u0 = i == 1 ? [x[1]] : i == 2 ? solutions[1].u[end] : [x[4]]
        @test sol.prob.u0 == expected_u0
        @test isapprox(sol.u[end][1], y[i]; atol = 1.0e-9, rtol = 1.0e-9)
    end
    @test isapprox(
        sequential_loss(case, x; solve_kwargs), endpoint_loss(y);
        atol = 1.0e-9, rtol = 1.0e-9
    )
    return nothing
end

function test_value_and_gradient(case, x, value, gradient; gradient_atol, gradient_rtol)
    expected_value, expected_gradient = reference_value_and_gradient(case, x)
    @test isapprox(value, expected_value; atol = 1.0e-9, rtol = 1.0e-9)
    @test size(gradient) == size(x)
    @test all(isfinite, gradient)
    for j in eachindex(x)
        @test isapprox(
            gradient[j], expected_gradient[j];
            atol = gradient_atol, rtol = gradient_rtol
        )
    end
    return nothing
end

"""
    test_sequential_ad(backend; solve_kwargs=(;), gradient_atol=1e-7, gradient_rtol=1e-6)

Run the same three-stage linear and nonlinear gradient contracts, each with
out-of-place and in-place dynamics, for any DI backend. Load the backend's
packages before calling. `solve_kwargs` can supply
a backend-specific SciML `sensealg`; preserve the endpoint-only saving options.

Unavailable backends are explicitly skipped. Differentiation errors from an
available backend are test errors, never converted into skips or broken tests.
"""
function test_sequential_ad(
        backend; solve_kwargs = (;),
        gradient_atol = 1.0e-7, gradient_rtol = 1.0e-6
    )
    return @testset "Sequential AD: $(repr(backend))" begin
        if !DI.check_available(backend)
            @test_skip DI.check_available(backend)
        else
            for case in CASES
                @testset "$(case_name(case)) / $(form_name(case))" begin
                    loss = x -> sequential_loss(case, x; solve_kwargs)
                    @testset "primal" begin
                        for x in INPUTS
                            test_primal(case, x; solve_kwargs)
                        end
                    end
                    @testset "unprepared" begin
                        for x in INPUTS
                            unchanged = copy(x)
                            value, gradient = DI.value_and_gradient(loss, backend, x)
                            test_value_and_gradient(
                                case, x, value, gradient;
                                gradient_atol, gradient_rtol
                            )
                            @test x == unchanged
                        end
                    end
                    @testset "prepared, changed input, repeated evaluation" begin
                        prep = DI.prepare_gradient(loss, backend, first(INPUTS))
                        for x in (INPUTS..., first(INPUTS))
                            unchanged = copy(x)
                            value, gradient = DI.value_and_gradient(loss, prep, backend, x)
                            test_value_and_gradient(
                                case, x, value, gradient;
                                gradient_atol, gradient_rtol
                            )
                            @test x == unchanged
                        end
                    end
                end
            end
        end
    end
end

end # module SequentialADTests
