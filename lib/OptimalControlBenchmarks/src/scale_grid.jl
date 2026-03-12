function scale_grids!(tspan::Tuple{Float64, Float64}, grids::BenchmarkGrids)
    scale = last(tspan) - first(tspan)
    shift = first(tspan)

    grids.control_grid .*= scale .+ shift
    grids.shooting_grid .*= scale .+ shift
    grids.constraint_grid .*= scale .+ shift
    return grids
end
