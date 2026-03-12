using BenchmarkTools
using OptimalControlBenchmarks

function run_all(benchmarks, optimizer, grids)

    results = []

    for prob in benchmarks

        # println("Running ", prob.name)
        sol = solve_with_corleone(
            prob,
            optimizer,
	    grids
        )
        println(sol)
        t = 0  # @belapsed solve_with_corleone($prob)

        push!(results, (
            name = "",  # prob.name,
            time = t
        ))

    end

    return results
end
