module CorleoneInverseCairoMakieExt

import CorleoneInverse
using CairoMakie: Axis, Figure, hlines!, scatter!, vlines!

function _parameter_index(names, parameter)
    parameter isa Integer && return parameter
    index = findfirst(==(string(parameter)), names)
    isnothing(index) && throw(ArgumentError("Unknown parameter $parameter."))
    return index
end

"""
    plot_beale_pairs(result; pairs=nothing, show_sampling_box=true, columns=2)

Plot accepted Beale confidence-region samples pairwise. `pairs` may contain
parameter names or indices; by default all pairs are plotted. The sampling
limits are shown as dashed red lines, and the estimate as a black star.
Requires `CairoMakie` to be loaded.
"""
function CorleoneInverse.plot_beale_pairs(
        result; pairs = nothing, show_sampling_box = true, columns::Integer = 2
    )
    columns > 0 || throw(ArgumentError("columns must be positive."))
    names = result.names
    samples = result.parameters
    estimate = result.estimate
    p = length(names)
    indices = isnothing(pairs) ? [(i, j) for i in 1:p for j in (i + 1):p] :
        [(_parameter_index(names, a), _parameter_index(names, b)) for (a, b) in pairs]
    isempty(indices) && throw(ArgumentError("No parameter pairs to plot."))
    all(pair -> 1 <= pair[1] <= p && 1 <= pair[2] <= p && pair[1] != pair[2], indices) ||
        throw(ArgumentError("Parameter pair indices must be distinct and in 1:$p."))

    rows = cld(length(indices), columns)
    fig = Figure(size = (420 * min(columns, length(indices)), 380 * rows))
    for (k, (i, j)) in enumerate(indices)
        ax = Axis(fig[cld(k, columns), mod1(k, columns)], xlabel = names[i], ylabel = names[j])
        scatter!(ax, samples[i, :], samples[j, :]; markersize = 5, color = :steelblue)
        scatter!(ax, [estimate[i]], [estimate[j]]; marker = :star5, markersize = 18, color = :black)
        if show_sampling_box
            vlines!(ax, [result.lower[i], result.upper[i]]; color = :red, linestyle = :dash)
            hlines!(ax, [result.lower[j], result.upper[j]]; color = :red, linestyle = :dash)
        end
    end
    return fig
end

end
