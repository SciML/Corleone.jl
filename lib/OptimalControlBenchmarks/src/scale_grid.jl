function validate_normalized_grids(tspan, grids)
    lo, hi = tspan
    isfinite(lo) && isfinite(hi) && lo < hi || throw(ArgumentError("The time span must be finite and increasing."))
    for grid in (grids.control_grid, grids.shooting_grid, grids.constraint_grid)
        issorted(grid) && allunique(grid) && all(t -> isfinite(t) && 0 <= t <= 1, grid) ||
            throw(ArgumentError("Benchmark grids must be sorted, unique, and contained in [0, 1]."))
    end
    return nothing
end

"""
$(SIGNATURES)

Map normalized grids into `tspan` with `t = t0 + (tf - t0) * τ`, updating their
existing buffers. Validate every grid before writing. Problem constructors use
[`scale_grids`](@ref) to keep the caller's normalized configuration reusable.
"""
function scale_grids!(tspan::Tuple{<:Real, <:Real}, grids::BenchmarkGrids)
    validate_normalized_grids(tspan, grids)
    lo, hi = tspan
    for grid in (grids.control_grid, grids.shooting_grid, grids.constraint_grid)
        grid .= lo .+ (hi - lo) .* grid
    end
    return grids
end

"""
$(SIGNATURES)

Return independent grids scaled from `[0, 1]` into `tspan`. The input grids and
all their buffers remain unchanged, so one configuration can be reused across
problems with different horizons.
"""
function scale_grids(tspan::Tuple{<:Real, <:Real}, grids::BenchmarkGrids)
    scaled = BenchmarkGrids(copy(grids.control_grid), copy(grids.shooting_grid), copy(grids.constraint_grid))
    return scale_grids!(tspan, scaled)
end
