function load_benchmarks()
    benchmarks = []
    problems_dir = joinpath(@__DIR__, "problems")

    for f in readdir(problems_dir)
        # Skip non-julia files
        endswith(f, ".jl") || continue
        
        # Get the path and the expected function name (stripping .jl)
        path = joinpath(problems_dir, f)
        func_name_sym = Symbol(splitext(f)[1])

        # Evaluate the file content into the current module
        include(path)

        # Retrieve the function object by its name
        try
            # Look up the symbol in the current module's scope
            func_obj = getfield(@__MODULE__, func_name_sym)
            
            if func_obj isa Function
                push!(benchmarks, func_obj)
            else
                @warn "File $f included, but $func_name_sym is not a function."
            end
        catch e
            @error "Failed to load function $func_name_sym from $f" exception=e
        end
    end

    return benchmarks
end
