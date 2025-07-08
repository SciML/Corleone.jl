using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(path = joinpath(@__DIR__, ".."))

include("utils.jl")

@info "Extract tests from examples"

const TESTPATH = joinpath(
    pwd(), "..", "test", "examples"
)


const EXAMPLEDOCPATH = joinpath(
    pwd(), "examples"
)

const EXAMPLEPATH = joinpath(
    EXAMPLEDOCPATH, "code"
)

# Find all examples 
for file in readdir(EXAMPLEDOCPATH)
    path_ = splitpath(file)
    finfo = split(path_[end], ".")
    if last(finfo) == "qmd"
        @info " Processing $(file)..."
        # Export raw code to /examples 
        if parse_example_file(joinpath(EXAMPLEDOCPATH, file), joinpath(EXAMPLEPATH, first(finfo) * ".jl"))
            @info "   Example ✔"
        end
        # Export raw code to 
        if parse_example_file_to_test(joinpath(EXAMPLEDOCPATH, file), joinpath(TESTPATH, first(finfo) * ".jl"))
            @info "   Testfile ✔"
        end
    end
end

@info "Finished"
