@concrete terse struct ShootingLayer{L <: ParallelLayer} <: LuxCore.AbstractLuxWrapperLayer{(:layer)}
    layer::L
end

function ShootingLayer(
    problem::SciMLBase.AbstractDEProblem, 
    variable_id,
    controls...; 
    kwargs...
    )
    layer = ParallelLayer(problem, variable_id, controls...; kwargs...)
    ShootingLayer(layer)
end

function (layer::ShootingLayer)(x, ps, st)
    ret, st = layer.layer(x, ps, st)
    Solutions.Trajectory(ret, layer.layer.control_cache)
end