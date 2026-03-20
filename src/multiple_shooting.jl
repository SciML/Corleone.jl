"""
$(TYPEDEF)

Defines a layer for multiple shooting. Simply a wrapper for the [ParallelShootingLayer](@ref) but returns a single trajectory.
"""
struct MultipleShootingLayer{L, S <: NamedTuple} <: LuxCore.AbstractLuxWrapperLayer{:layer}
    "The instance of a [ParallelShootingLayer](@ref) to be solved in parallel."
    layer::L
    "Indicator for shooting constraints for each of the layers."
    shooting_variables::S
end

function MultipleShootingLayer(layer::LuxCore.AbstractLuxLayer, shooting_points::Real...; kwargs...)
    tspan = get_tspan(layer)
    tpoints = unique!(sort!(vcat(collect(shooting_points), collect(tspan))))
    layers = ntuple(
        i -> remake(
            layer,
            tspan = (tpoints[i], tpoints[i + 1]),
            tunable_ic = get_tunable_u0(layer, i != 1),
        ), length(tpoints) - 1
    )
    layers = NamedTuple{ntuple(i -> Symbol(:layer_, i), length(layers))}(layers)
    shooting_variables = map(get_shooting_variables, layers)
    layer = ParallelShootingLayer(layers; kwargs...)

    return MultipleShootingLayer{typeof(layer), typeof(shooting_variables)}(layer, shooting_variables)
end

get_problem(layer::MultipleShootingLayer) = get_problem(layer.layer.layers[1])
get_quadrature_indices(layer::MultipleShootingLayer) = get_quadrature_indices(layer.layer.layers[1])
get_tspan(layer::MultipleShootingLayer) = begin
    t0, _ = get_problem(layer.layer.layers[1]).tspan
    _, tinf = get_problem(layer.layer.layers[end]).tspan
    (t0, tinf)
end
get_timegrid(layer::MultipleShootingLayer) = reduce(vcat, values(get_timegrid(layer.layer)))

function SciMLBase.remake(layer::MultipleShootingLayer; kwargs...)
    newlayer = remake(layer.layer; kwargs...)
    return MultipleShootingLayer{typeof(newlayer), typeof(layer.shooting_variables)}(newlayer, layer.shooting_variables)
end

function (layer::MultipleShootingLayer)(u0, ps, st)
    results, st = layer.layer(u0, ps, st)
    return Trajectory(layer, results), st
end

function get_number_of_shooting_constraints(layer::MultipleShootingLayer)
    # We ignore the first shooting variables here
    return size(reduce(vcat, fleaves(Base.tail(layer.shooting_variables))), 1)
end

function matchings(layer::MultipleShootingLayer, us, sub_trajs)
    (; shooting_variables) = layer
    vars = variable_symbols(get_problem(layer))
    return map(Base.OneTo(length(shooting_variables) - 1)) do i
        specs = shooting_variables[i + 1]
        state_matching = map(specs.state) do id
            Symbol(vars[id]), first(us[i + 1])[id] .- last(us[i])[id]
        end |> NamedTuple
        control_matching = map(specs.control) do csym
            traj_prev = sub_trajs[i]
            traj_next = sub_trajs[i + 1]
            v_prev = getproperty(_apply(traj_prev.controls.model, last(traj_prev.t),  traj_prev.controls.ps, traj_prev.controls.st)[1], csym)
            v_next = getproperty(_apply(traj_next.controls.model, first(traj_next.t), traj_next.controls.ps, traj_next.controls.st)[1], csym)
            Symbol(csym), v_next .- v_prev
        end |> NamedTuple
        Symbol(:matching_, i), (; state = state_matching, control = control_matching)
    end |> NamedTuple
end

function Trajectory(
        layer::MultipleShootingLayer, solutions::NamedTuple{fields};
        kwargs...
    ) where {fields}

    sub_trajs = values(solutions)
    us = map(Base.Fix2(getproperty, :u), sub_trajs)
    ts = map(Base.Fix2(getproperty, :t), sub_trajs)
    shooting_violations = matchings(layer, us, sub_trajs)
    # Use the first sub-trajectory's StatefulLuxLayer for the combined controls
    controls = first(sub_trajs).controls
    p = deepcopy(first(sub_trajs).p)
    # Update the quadratures
    quadratures = get_quadrature_indices(layer)
    q_prev = last(us[1])
    keeper = [i in quadratures for i in eachindex(q_prev)]
    us_ = map(us[2:end]) do ui
        new_uij = map(uij -> uij .+ keeper .* q_prev, ui)
        q_prev = keeper .* last(new_uij)
        new_uij
    end
    us = [us[1], us_...]
    unew = reduce(
        vcat, map(i -> i == lastindex(us) ? us[i] : us[i][1:(end - 1)], eachindex(us))
    )
    t_new = reduce(
        vcat, map(i -> i == lastindex(ts) ? ts[i] : ts[i][1:(end - 1)], eachindex(ts))
    )
    sys = first(solutions).sys
    return Trajectory{typeof(sys), typeof(unew), typeof(p), typeof(t_new), typeof(controls), typeof(shooting_violations)}(sys, unew, p, t_new, controls, shooting_violations)
end
