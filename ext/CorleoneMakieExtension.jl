module CorleoneMakieExtension
using Corleone
using Makie

Makie.plottype(sol::Trajectory) = Makie.Lines

function Makie.used_attributes(::Type{<:Plot}, sol::Trajectory)
    return (:vars, :idxs)
end

function Makie.convert_arguments(
        PT::Type{<:Plot},
        sol::Trajectory;
        vars = nothing,
        idxs = nothing
    )

    if !isnothing(vars)
        (!isnothing(idxs)) && error("Can't simultaneously provide vars and idxs!")
        idxs = vars
    end

    idxs = isnothing(idxs) ? eachindex(sol.u[1]) : idxs
    idxs = eltype(idxs) == Symbol ? [sol.sys.variables[i] for i in idxs] : idxs

    plot_vecs = reduce(hcat, sol.u)[idxs, :]

    plot_type_sym = Makie.plotsym(PT)

    inv_sys_variables = Dict(value => key for (key, value) in sol.sys.variables)
    labels = string.([inv_sys_variables[i] for i in idxs])

    return map(
        (x, y, label, i) -> PlotSpec(plot_type_sym, Point2f.(x, y); label, color = Cycled(i)),
        [sol.t for _ in eachindex(idxs)],
        eachrow(plot_vecs),
        labels,
        eachindex(idxs)
    )
end
end
