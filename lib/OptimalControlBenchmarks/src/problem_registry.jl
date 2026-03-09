function load_benchmarks()

    benchmarks = []

    for f in readdir(joinpath(@__DIR__,"problems"))
        endswith(f,".jl") || continue

        modname = Symbol(splitext(f)[1])

        include(joinpath(@__DIR__,"problems",f))

        mod = getfield(@__MODULE__, modname)

        push!(benchmarks, mod.benchmark)
    end

    benchmarks
end