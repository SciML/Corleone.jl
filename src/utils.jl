"""
Extend 'extend' with additional arguments.
"""
function extend_system(newsys, basesys; name=nameof(basesys),
    description=ModelingToolkit.get_description(basesys),
    gui_metadata=ModelingToolkit.get_gui_metadata(basesys),
    costs=ModelingToolkit.get_costs(basesys), consolidate=ModelingToolkit.get_consolidate(basesys),
    constraintsystem=ModelingToolkit.constraints(basesys),
    kwargs...)
    newsys = extend(newsys, basesys; name, description, gui_metadata)
    newsys = isnothing(costs) ? newsys : @set newsys.costs = costs
    newsys = isnothing(consolidate) ? newsys : @set newsys.consolidate = consolidate
    newsys = isnothing(constraintsystem) ? newsys : @set newsys.constraints = constraintsystem
    return newsys
end

function __fallbackdefault(x)
    if ModelingToolkit.hasbounds(x)
        lo, hi = ModelingToolkit.getbounds(x)
        return (hi - lo) / 2
    else
        Symbolics.hasmetadata(x, Symbolics.VariableDefaultValue)
        return Symbolics.getdefaultval(x)
    end
    return zero(Symbolics.symtype(x))
end

function __default_shooting_vars(prob::SciMLBase.AbstractSciMLProblem)
    psyms = SciMLBase.getparamsyms(prob)
    return filter(is_shootingvariable, psyms)
end

function __shooting_timepoints(prob::SciMLBase.AbstractSciMLProblem)
    psyms = SciMLBase.getparamsyms(prob)
    shooting_points = filter(is_shootingpoint, psyms)
    unique!(sort!(reduce(vcat, Symbolics.getdefaultval.(shooting_points))))
end
