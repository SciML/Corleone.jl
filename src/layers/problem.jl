struct DynamicProblemLayer{S <: ShootingLayer, O, C} <: LuxCore.AbstractLuxContainerLayer{(:shooting, :objective, :constraints)}
    shooting::S
    objective::O
    constraints::C
end

function DynamicProblemLayer(layer::ShootingLayer, obj, constraints...; kwargs...)
    objective = DynamicFunctionLayer(layer, obj; kwargs...)
    constraints = if !isempty(constraints) 
        DynamicFunctionLayer(layer, constraints...; kwargs...)
    else
        nothing
    end
    return DynamicProblemLayer{typeof(layer), typeof(objective), typeof(constraints)}(layer, objective, constraints)
end

function evaluate_objective(prob::DynamicProblemLayer, x::SciMLBase.AbstractDEProblem, ps, st)
    x = remake(x, saveat = st.objective.saveat)
    traj, shooting_st = prob.shooting(x, ps.shooting, st.shooting)
    obj, objective_st = prob.objective(traj, ps.objective, st.objective)
    return obj, (; shooting = shooting_st, objective = objective_st, constraints = st.constraints)
end

evaluate_constraints(::DynamicProblemLayer{<:ShootingLayer, <:Any, Nothing}, ::SciMLBase.AbstractDEProblem, ps, st) = nothing, st
evaluate_constraints(::DynamicProblemLayer{<:ShootingLayer, <:Any, Nothing}, ::Tuple{<:Any, <:SciMLBase.AbstractDEProblem}, ps, st) = nothing, st

function evaluate_constraints(prob::DynamicProblemLayer, x::SciMLBase.AbstractDEProblem, ps, st)
    x = remake(x, saveat = st.constraints.saveat)
    traj, shooting_st = prob.shooting(x, ps.shooting, st.shooting)
    cons, cons_st = prob.constraints(traj, ps.constraints, st.constraints)
    return cons, (; shooting = shooting_st, objective = st.objective, constraints = cons_st)
end

function evaluate_constraints(prob::DynamicProblemLayer, (res, x)::Tuple{<:Any, <:SciMLBase.AbstractDEProblem}, ps, st)
    x = remake(x, saveat = st.constraints.saveat)
    traj, shooting_st = prob.shooting(x, ps.shooting, st.shooting)
    cons, cons_st = prob.constraints(res, traj, ps.constraints, st.constraints)
    return cons, (; shooting = shooting_st, objective = st.objective, constraints = cons_st)
end

function predict(prob::DynamicProblemLayer, x::SciMLBase.AbstractDEProblem, ps, st)
    T = eltype(x.tspan)
    saveat = vcat(
        get(x.kwargs, :saveat, T[]),
        get(st.objective, :saveat, T[]),
        get(st.constraints, :saveat, T[]),
    )
    x = remake(x, saveat = (unique! ∘ sort!)(saveat), tspan = extrema(saveat))
    traj, shooting_st = prob.shooting(x, ps.shooting, st.shooting)
    traj, merge(st, (; shooting = shooting_st))
end

function (prob::DynamicProblemLayer)(x, ps, st)
    obj, st = evaluate_objective(prob, x, ps, st)
    cons, st = evaluate_constraints(prob, x, ps, st)
    (obj, cons), st
end
