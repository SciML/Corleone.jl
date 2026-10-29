"""
$(SIGNATURES)

Build a prepared Corleone model from a benchmark constructor and normalized grids.
Control grids contain interval starts; a final endpoint is discarded. Shooting
endpoints are accepted and only interior points create matching resets. Pass an
empty shooting grid for single shooting. The benchmark's explicit time span is
used even when a grid omits its final endpoint.

`algorithm` selects the differential-equation solver. Other keywords are forwarded
to `CorleoneDynamicOptProblem`, including `problem_kwargs`, `ensemblealg`, and `rng`.
The returned model retains its initialized parameters and state.
"""
function prepare_benchmark(benchmark, grids::BenchmarkGrids; algorithm = Tsit5(), kwargs...)
    data = benchmark(grids)
    tspan = get(data, :tspan, nothing)
    control_grid = isnothing(tspan) ? data.grids.control_grid :
        filter(t -> t < last(tspan), data.grids.control_grid)
    controls = [control => control_grid for control in inputs(data.system)]
    return CorleoneDynamicOptProblem(
        data.system, Pair[], controls...;
        algorithm, shooting = data.grids.shooting_grid, tspan, kwargs...
    )
end

"""
$(SIGNATURES)

Prepare and solve a benchmark with Corleone's SciMLBase optimization bridge.
`plot_res=true` displays the resulting states, controls, and quadratures. Reconstruct
optimizer values with the selected vectorizer and reuse the prepared model's state.

`AD` defaults to `AutoForwardDiff()`. `vectorize` defaults to Corleone's loaded
vectorizer. `model_kwargs` configure `prepare_benchmark`; remaining keywords
configure the optimization solver and override the default iteration limit,
tolerance, or limited-memory Hessian approximation. Return the optimization solution,
including its actual solver status; convergence is not assumed.
"""
function solve_with_corleone(
        benchmark, optimizer, grids::BenchmarkGrids, plot_res::Bool = true;
        algorithm = Tsit5(), AD = AutoForwardDiff(), vectorize = default_vectorizer(),
        model_kwargs = (;), kwargs...
    )
    model = prepare_benchmark(benchmark, grids; algorithm, model_kwargs...)
    optprob = OptimizationProblem(model, AD; vectorize)
    options = (; max_iter = 100, tol = 5.0e-6, hessian_approximation = "limited-memory", kwargs...)
    solution = solve(optprob, optimizer; options...)
    if plot_res
        _, reconstruct = pack(vectorize, model.ps)
        trajectory, _ = model.layer(
            (model.layer.problem, model.evaluation_grid), reconstruct(solution.u), model.st
        )
        display(plot_oc_problem(trajectory))
    end
    return solution
end
