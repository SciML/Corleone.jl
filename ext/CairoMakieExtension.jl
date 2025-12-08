module CairoMakieExtension
using Corleone
using CairoMakie

Makie.plottype(sol::Trajectory) = Makie.Lines

function Makie.used_attributes(::Type{<:Plot}, sol::Trajectory)
    (:vars, :idxs)
end

function CairoMakie.convert_arguments(PT::Type{<:Plot},
            sol::Trajectory;
            vars=nothing,
            idxs=nothing)

    if !isnothing(vars)
        (!isnothing(idxs)) && error("Can't simultaneously provide vars and idxs!")
        idxs = vars
    end

    idxs = isnothing(idxs) ? eachindex(sol.u[1]) : idxs
    idxs = eltype(idxs) == Symbol ? [sol.sys.variables[i] for i in idxs] : idxs

    plot_vecs = reduce(hcat, sol.u)[idxs,:]

    plot_type_sym = Makie.plotsym(PT)

    inv_sys_variables = Dict(value => key for (key,value) in sol.sys.variables)
    labels = string.([inv_sys_variables[i] for i in idxs])

    map((x, y, label) -> PlotSpec(plot_type_sym, Point2f.(x, y); label),
            [sol.t for _ in eachindex(idxs)],
            eachrow(plot_vecs),
            labels)
    end
end