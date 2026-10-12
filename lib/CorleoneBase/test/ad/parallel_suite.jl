module ParallelADTests

using CorleoneBase
import CommonSolve
import DifferentiationInterface as DI
using OrdinaryDiffEqTsit5: Tsit5
using SciMLBase: ODEProblem, remake, isinplace, successful_retcode
using Test

include("fixtures.jl")

# Reuse the sequential models and their independent hand-derived references.
# Each trajectory depends on every input, with distinct states and parameters.
trajectory_input(x, i) = (1 + i / 10) .* x .+ i / 20

function execution_locations(mode)
    probe = ParallelProblem(ODEProblem((u, p, t) -> -p .* u, [1.0], (0.0, 1.0), 0.4))
    output = (sol, ctx) -> ((getpid(), Threads.threadid()), false)
    return CommonSolve.solve(probe, Tsit5(), mode; trajectories = 12, output_func = output).u
end

function trajectory_problem(case, x, wrapped)
    ode = ODEProblem(dynamics(case), [x[1]], (0.0, 1.0), [x[2]])
    # Out-of-place problems solved directly by the ensemble machinery need the
    # state and derivative to share one array representation, like the
    # per-stage preparation the sequential solve applies internally. Plain
    # vectors, scalars, and functions pass through unchanged, so this is a
    # no-op for untraced inputs and leaves ForwardDiff coverage unaffected.
    wrapped || return CorleoneBase.prepare_stage_problem(ode)
    transition = (sol, stage) -> stage == 2 ?
        remake(sol.prob; u0 = sol.u[end], p = [x[3]], tspan = (1.0, 2.0)) :
        remake(sol.prob; u0 = [x[4]], tspan = (2.0, 3.0))
    return SequentialProblem(ode; transition, terminal = (sol, i) -> i >= 3)
end

function ensemble_solutions(case, x, wrapped, mode; solve_kwargs = (;))
    template = trajectory_problem(case, x, wrapped)
    hook = (problem, template, ctx) ->
    trajectory_problem(case, trajectory_input(x, ctx.sim_id), wrapped)
    problem = ParallelProblem(template; prob_func = hook)
    return CommonSolve.solve(
        problem, Tsit5(), mode;
        trajectories = 3, merge(SOLVE_KWARGS, solve_kwargs)...
    )
end

function ensemble_loss(case, x, wrapped, mode; solve_kwargs = (;))
    ensemble = ensemble_solutions(case, x, wrapped, mode; solve_kwargs)
    return sum(
        i * (
            wrapped ? endpoint_loss(ntuple(k -> ensemble.u[i][k].u[end][1], 3)) :
                (ensemble.u[i].u[end][1] - TARGETS[1])^2 / 2
        ) for i in 1:3
    )
end

function ensemble_reference(case, x, wrapped)
    value = 0.0
    gradient = zeros(length(x))
    for i in 1:3
        z = trajectory_input(x, i)
        y, jac = reference_endpoints(case, wrapped ? z : [z[1], z[2], 0.0, 0.0])
        for k in 1:(wrapped ? 3 : 1)
            weight = i * WEIGHTS[k]
            value += weight * (y[k] - TARGETS[k])^2 / 2
            gradient .+= weight * (y[k] - TARGETS[k]) .* jac[k, 1:length(x)] .* (1 + i / 10)
        end
    end
    return value, gradient
end

function check_gradient(case, x, wrapped, value, gradient; gradient_atol, gradient_rtol)
    expected_value, expected_gradient = ensemble_reference(case, x, wrapped)
    @test isapprox(value, expected_value; atol = 1.0e-9, rtol = 1.0e-9)
    @test size(gradient) == size(x)
    @test all(isfinite, gradient)
    for j in eachindex(x)
        @test isapprox(gradient[j], expected_gradient[j]; atol = gradient_atol, rtol = gradient_rtol)
    end
    return
end

"""
    test_parallel_ad(backend, modes; solve_kwargs=(;), gradient_atol=1e-7, gradient_rtol=1e-6)

Reusable ensemble AD contracts. Only explicitly supplied backends are exercised;
an unavailable backend or any differentiation failure is an error, not a skip.
Workers must load this module, the backend, and any sensitivity configuration
before distributed evaluation. Tape-based reverse-mode backends such as
AutoReverseDiff trace the inner solves directly and require the pass-through
sensitivity algorithm and `safetycopy = false` (like the sequential suite's
pass-through; SciMLBase's default per-simulation template deepcopy recurses
through the ReverseDiff tape graph until the stack overflows on Julia 1.10 and
1.11) via `solve_kwargs`; compiled tapes stay out of scope.
AutoMooncake uses the explicitly loaded CorleoneBaseMooncakeExtension and
`safetycopy = false`. Its rule runs the requested-mode primal ensemble, then
recomputes and differentiates the actual deterministic trajectory solves in
serial; it does not differentiate scheduler, RNG, or progress infrastructure.
AutoMooncakeForward shares that extension: its frule runs the genuine
requested-mode primal (so threaded and distributed execution is real) and
returns the serial replay's forward-mode tangent, computed from the actual
trajectory solves with the incoming Duals.
The full matrix and all contracts below are shared unchanged by every backend.
"""
function test_parallel_ad(backend, modes; solve_kwargs = (;), gradient_atol = 1.0e-7, gradient_rtol = 1.0e-6)
    @test DI.check_available(backend)
    return @testset "Parallel AD: $(repr(backend))" begin
        for mode in modes, wrapped in (false, true), case in CASES
            @testset "$(typeof(mode)) / $(wrapped ? "sequential" : "ODE") / $(case_name(case)) / $(form_name(case))" begin
                inputs = wrapped ? INPUTS : map(x -> x[1:2], INPUTS)
                loss = x -> ensemble_loss(case, x, wrapped, mode; solve_kwargs)
                for x in inputs
                    unchanged = copy(x)
                    ensemble = ensemble_solutions(case, x, wrapped, mode; solve_kwargs)
                    @test length(ensemble.u) == 3
                    for i in 1:3
                        z = trajectory_input(x, i)
                        y, _ = reference_endpoints(case, wrapped ? z : [z[1], z[2], 0.0, 0.0])
                        stages = wrapped ? ensemble.u[i] : (ensemble.u[i],)
                        @test length(stages) == (wrapped ? 3 : 1)
                        for (k, sol) in enumerate(stages)
                            @test successful_retcode(sol)
                            @test isinplace(sol.prob) == is_inplace(case)
                            @test sol.prob.tspan == (Float64(k - 1), Float64(k))
                            @test sol.t == [Float64(k - 1), Float64(k)]
                            @test sol.prob.p == [z[k == 1 ? 2 : 3]]
                            @test sol.prob.u0 == (k == 1 ? [z[1]] : k == 2 ? stages[1].u[end] : [z[4]])
                            @test isapprox(sol.u[end][1], y[k]; atol = 1.0e-9, rtol = 1.0e-9)
                        end
                    end
                    @test isapprox(loss(x), first(ensemble_reference(case, x, wrapped)); atol = 1.0e-9, rtol = 1.0e-9)
                    @test x == unchanged
                end
                @testset "unprepared" begin
                    for x in inputs
                        unchanged = copy(x)
                        value, gradient = DI.value_and_gradient(loss, backend, x)
                        check_gradient(case, x, wrapped, value, gradient; gradient_atol, gradient_rtol)
                        @test x == unchanged
                    end
                end
                @testset "prepared, changed input, repeated evaluation" begin
                    original = copy(first(inputs))
                    prep = DI.prepare_gradient(loss, backend, first(inputs))
                    @test first(inputs) == original
                    for x in (inputs..., first(inputs))
                        unchanged = copy(x)
                        value, gradient = DI.value_and_gradient(loss, prep, backend, x)
                        check_gradient(case, x, wrapped, value, gradient; gradient_atol, gradient_rtol)
                        @test x == unchanged
                    end
                end
            end
        end
    end
end

end
