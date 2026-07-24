module CorleoneMakieExtension
using Corleone.Solutions: ControlSegment, AbstractCompositeSolution, control_indices, quadrature_indices
using Makie
using SymbolicIndexingInterface

Makie.plottype(::AbstractCompositeSolution) = Makie.Lines

function Makie.used_attributes(::Type{<:Plot}, sol::AbstractCompositeSolution)
    return (:vars, :idxs, :show_segments, :show_segment_ends)
end

function Makie.convert_arguments(
        PT::Type{<:Plot},
        sol::AbstractCompositeSolution;
        vars = [],
        idxs = Int[],
        show_segments::Bool = true,
        show_segment_ends::Bool = true,
        kwargs...
    )
    new_idxs = reduce(
        vcat, map(vars) do vi
            SymbolicIndexingInterface.variable_index(sol, vi)
        end
    )

    append!(idxs, new_idxs)
    cidx = control_indices(sol.sys)
    qidx = quadrature_indices(sol.sys)
    control_idx = intersect(idxs, cidx)
    labels = string.(variable_symbols(sol))
    colors = [Cycled(i) for i in eachindex(labels)]
    vis = reduce(hcat, state_values(sol, idxs))
    t = current_time(sol)
    plot_type_sym = Makie.plotsym(PT)
    plots = map(
        ((id, i)::Tuple) -> if i ∈ control_idx
            PlotSpec(Makie.Stairs, Point2f.(t, view(vis, id, :)); label = labels[i], color = colors[i], step = :post)
        else
            PlotSpec(plot_type_sym, Point2f.(t, view(vis, id, :)); label = labels[i], color = colors[i])
        end, enumerate(idxs)
    )

    if show_segments && length(sol.segments) > 1
        tends = collect(map(xi -> current_time(xi)[end], sol.segments[1:(end - 1)]))
        push!(plots, PlotSpec(VLines, tends; color = :black, linestyle = :dash))
    end

    if show_segment_ends && length(sol.segments) > 1
        foreach(sol.segments[1:(end - 1)]) do seg
            ti = current_time(seg)[end]
            ui = state_values(seg)[end]
            for i in setdiff(idxs, qidx)
                push!(
                    plots,
                    PlotSpec(
                        Scatter, Point2f(ti, ui[i]); marker = :xcross, color = colors[i]
                    )
                )
            end
        end
    end
    return plots
end
end
