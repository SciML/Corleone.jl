module CorleoneMakieExtension
using Corleone
using SymbolicIndexingInterface
using Makie

Makie.plottype(sol::Trajectory) = Makie.Lines

function Makie.used_attributes(::Type{<:Plot}, sol::Trajectory)
    return (:vars, :idxs)
end

maybevec(x::AbstractArray) = eachrow(reduce(hcat, x))
maybevec(x) = x

_getindex(x, i) = getindex(x, i)
_getindex(x::Symbol, i) = x

function Makie.convert_arguments(
        PT::Type{<:Plot},
        sol::Trajectory;
        idxs::AbstractVector{<:Int} = Int64[],
        vars::AbstractVector = [],
        kwargs...
    )
    if !isempty(idxs)
        append!(
            vars,
            variable_symbols(sol)[idxs]
        )
    end
    if isempty(vars)
        for v in variable_symbols(sol)
            push!(vars, variable_index(sol, v))
        end
    end
    ts = []
    xs = []
    labels = String[]
    foreach(vars) do var
        if is_timeseries_parameter(sol, var)
            x_current = maybevec(getp(sol, var)(sol))
            append!(xs, x_current)
            for i in eachindex(x_current)
                push!(ts, sol.controls.collection[1].t)
                push!(labels, string(_getindex(var, i)))
            end
        else
            x_current = maybevec(getsym(sol, var)(sol))
            append!(xs, x_current)
            for i in eachindex(x_current)
                push!(ts, sol.t)
                push!(labels, string(_getindex(var, i)))
            end
        end
    end

    plot_type_sym = Makie.plotsym(PT)

    return map(
        (x, y, label, i) -> PlotSpec(plot_type_sym, Point2f.(x, y); label, color = Cycled(i), kwargs...),
        ts,
        xs,
        labels,
        eachindex(labels)
    )
end
end
