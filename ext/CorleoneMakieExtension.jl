module CorleoneMakieExtension
using Corleone
import Corleone: trajectoryplot, trajectoryplot!
using SymbolicIndexingInterface
using Makie

@info "Loaded Makie extension"

Makie.plottype(::Trajectory) = Makie.PlotList

# Keep both sides of a reset without drawing a solid line between them.
function trajectory_points(times, values, reset_indices)
    points = Makie.Point2f[]
    sizehint!(points, length(times) + length(reset_indices))
    next_reset = 1
    for i in eachindex(times, values)
        if next_reset <= length(reset_indices) && i == reset_indices[next_reset]
            push!(points, Makie.Point2f(NaN, NaN))
            next_reset += 1
        end
        push!(points, Makie.Point2f(times[i], values[i]))
    end
    return points
end

# Residual vectors are compressed by active variable and indexed by solver segment.
# Use schedules to select the corresponding sample pairs instead.
function variable_reset_indices(sol::Trajectory, i)
    container = sol.layer.initialcondition
    group = findfirst(indices -> i in indices, container.indices)
    ic = values(container.initialconditions)[group]
    ic isa Corleone.InitialCondition || return Int[]
    if ic.parameter isa NamedTuple
        component = findfirst(==(i), container.indices[group])
        values(ic.parameter)[component] isa Corleone.FixedLayer && return Int[]
    end
    state = values(sol.st.initialcondition.initialconditions)[group]
    return filter(sol.reset_indices) do k
        haskey(ic.lookup, sol.t[k], nothing, state.lookup)
    end
end

function Makie.used_attributes(::Type{<:Plot}, ::Trajectory)
    return (:vars, :idxs, :show_segments, :show_variable_residuals, :patchcolor, :whiskerwidth)
end

function Makie.convert_arguments(
        PT::Type{<:Makie.Plot}, sol::Trajectory;
        vars = variable_symbols(symbolic_container(sol)),
        idxs = Int[],
        show_segments::Bool = !isempty(sol.reset_indices),
        show_variable_residuals::Bool = !isempty(sol.reset_indices),
        patchcolor = (:grey, 0.1),
        whiskerwidth = 0,
    )

    sys = symbolic_container(sol)
    varsyms = variable_symbols(sys)
    psyms = parameter_symbols(sys)
    # We assume idxs always refers to variables
    selected_variables = intersect(varsyms, collect(vars))
    selected_params = intersect(psyms, collect(vars))
    for i in idxs
        index = findfirst(xi -> variable_index(sys, xi) == i, varsyms)
        if index === nothing
            index = findfirst(xi -> parameter_index(sys, xi) == i, psyms)
            index !== nothing && append!(selected_params, index isa AbstractVector ? psyms[index] : psyms[[index]])
        else
            append!(selected_variables, index isa AbstractVector ? varsyms[index] : varsyms[[index]])
        end
    end

    times = current_time(sol)
    line_type = PT <: Makie.PlotList ? Makie.Lines : PT
    plots = Makie.PlotSpec[]
    color_index = 0


    if show_segments && !isempty(sol.reset_indices)
        idx = unique!(sort!(vcat(firstindex(times), sol.reset_indices, lastindex(times))))
        starts = times[idx[1:2:(end - 1)]]
        stops = times[idx[2:2:end]]
        push!(
            plots, Makie.PlotSpec(
                Makie.VSpan,
                starts, stops;
                color = patchcolor
            )
        )
    end

    foreach(selected_variables) do var
        points = trajectory_points(times, getindex(sol, var), sol.reset_indices)
        label = string(var)
        color = Makie.Cycled(color_index += 1)
        push!(plots, Makie.PlotSpec(line_type, points; label, color))
    end

    pcollection = get_parameter_timeseries_collection(sol)[1]

    foreach(selected_params) do var
        idx = parameter_index(sys, var)
        points = Makie.Point2f.(pcollection.t, map(Base.Fix2(getindex, idx), pcollection.u))
        label = string(var)
        color = Makie.Cycled(color_index += 1)
        push!(plots, Makie.PlotSpec(Makie.Stairs, points; label, color, step = :post))
    end


    if show_variable_residuals && !isempty(sol.reset_indices)
        # Early Makie 0.24 releases do not support line styles on range bars.
        style = Makie.is_attribute(Makie.Rangebars, :linestyle) ? (; linestyle = :dot) : (;)
        for (variable_color, var) in enumerate(selected_variables)
            i = variable_index(sys, var)
            indices = variable_reset_indices(sol, i)
            isempty(indices) && continue
            values = getindex(sol, var)
            before = values[indices .- 1]
            after = values[indices]
            color = Makie.Cycled(variable_color)
            push!(
                plots, Makie.PlotSpec(
                    Makie.Rangebars,
                    times[indices],
                    min.(before, after),
                    max.(before, after),
                    ;
                    color, label = "Δ$(var)", whiskerwidth = whiskerwidth, style...,
                )
            )
        end
    end

    return plots
end

function trajectoryplot!(
        container, solution; skips = [], show_legend = true,
        state_string = "u", control_string = "p", cost_string = "ℒₜ",
        kwargs...
    )
    sys = symbolic_container(solution)
    g = container isa Figure ? container[1, 1] : container
    axes = Axis[]

    variable_syms = variable_symbols(sys)
    quadrature_syms = quadrature_symbols(sys)
    state_syms = setdiff(variable_syms, quadrature_syms, skips)
    quadrature_syms = setdiff(quadrature_syms, skips)
    control_syms = setdiff(control_symbols(sys), skips)

    if !isempty(state_syms)
        i = length(axes) + 1
        push!(
            axes, Axis(g[i, 1], xlabel = "t", ylabel = state_string * "(t)")
        )
        current = last(axes)
        lines!(current, solution; vars = state_syms, kwargs...)
        if show_legend
            g[i, 2] = Legend(g, current, "States", framevisible = false)
        end
    end

    if !isempty(control_syms)
        i = length(axes) + 1
        push!(
            axes, Axis(g[i, 1], xlabel = "t", ylabel = control_string * "(t)")
        )
        current = last(axes)
        lines!(current, solution; vars = control_syms, kwargs...)
        if show_legend
            g[i, 2] = Legend(g, current, "Controls", framevisible = false)
        end
    end

    if !isempty(quadrature_syms)
        i = length(axes) + 1
        push!(
            axes, Axis(g[i, 1], xlabel = "t", ylabel = cost_string * "(t)")
        )
        current = last(axes)
        lines!(current, solution; vars = quadrature_syms, kwargs...)
        if show_legend
            g[i, 2] = Legend(g, current, "Acc. Costs", framevisible = false)
        end
    end

    linkxaxes!(axes...)

    return g
end

function trajectoryplot(sol::Trajectory; figure = (;), kwargs...)
    f = Figure(; figure...)
    trajectoryplot!(f[1, 1], sol; kwargs...)
    return f
end


end
