#using BenchmarkTools
using OptimalControlBenchmarks

function run_all(benchmarks)

    results = []

    for prob in benchmarks

        println("Running ", prob.name)
        sol = solve_with_corleone(prob)
        println(sol)
        #t = @belapsed solve_with_corleone($prob)

        push!(results, (
            name = prob.name,
            #time = t
        ))

    end

    return results
end