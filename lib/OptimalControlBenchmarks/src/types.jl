struct OptimalControlBenchmark
    name::Symbol
    description::String
    make_problem::Function
end

struct BenchmarkGrids
    control_grid::Vector{Float64}
    shooting_grid::Vector{Float64}
    constraint_grid::Vector{Float64}
end
