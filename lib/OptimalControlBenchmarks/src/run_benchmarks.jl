using BenchmarkTools

"""
    run_all(benchmarks, optimizer, grids)

Runs each benchmark problem with the Corleone benchmark solver.

# Arguments
- `benchmarks`: Iterable of benchmark constructor functions, such as the result of
  `load_benchmarks()`.
- `optimizer`: Optimization solver passed to `solve_with_corleone`.
- `grids`: Grid configuration passed to each benchmark constructor.

# Returns
A vector of named tuples containing each benchmark name and measured runtime placeholder.
"""
function run_all(benchmarks, optimizer, grids)

    results = []

    for prob in benchmarks
        name = nameof(prob)
        println("Running ", name)
        sol = solve_with_corleone(
            prob,
            optimizer,
            grids
        )
        println(sol)
        t = 0  # @belapsed solve_with_corleone($prob)

        push!(
            results, (
                name = name,
                time = t,
            )
        )

    end

    return results
end
