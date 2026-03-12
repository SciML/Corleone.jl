using BenchmarkTools
using OptimalControlBenchmarks

function run_all(benchmarks, optimizer, constraint_grid, control_grid, shooting_grid)

    results = []

    for prob in benchmarks

        println("Running ", prob.name)
        sol = solve_with_corleone(
            prob,
            optimizer,
            constraint_grid,
            control_grid,
            shooting_grid
        )
        println(sol)
        t = 0  # @belapsed solve_with_corleone($prob)

        push!(results, (
            name = prob.name,
            time = t
        ))

    end

    return results
end