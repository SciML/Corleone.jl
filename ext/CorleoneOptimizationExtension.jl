module CorleoneOptimizationExtension
using Corleone
using Optimization
using ComponentArrays
using LuxCore
using Random

function Optimization.OptimizationProblem(
        layer::Union{SingleShootingLayer, MultipleShootingLayer},
        loss::Union{Symbol, Expr};
        AD::Optimization.ADTypes.AbstractADType = AutoForwardDiff(),
        u0::ComponentVector = ComponentArray(first(LuxCore.setup(Random.default_rng(), layer))),
        constraints::Union{Nothing, <:Dict{Any, <:NamedTuple{(:t, :bounds)}}} = nothing,
        variable_type::Type{T} = Float64,
        kwargs...
    ) where {T}

    u0 = T.(u0)
    st = LuxCore.initialstates(Random.default_rng(), layer)

    if !isnothing(constraints)
        @assert all(collect(keys(constraints)) .|> typeof .<: Union{Expr, Symbol}) "Keys of the constraint dictionary need to be of type Symbol or Expr!"
    end

    # Bounds based on the variables
    lb, ub = Corleone.get_bounds(layer) .|> ComponentArray

    @assert all(lb .<= u0 .<= ub) "The initial variables are not within the bounds. Please check the input!"

    prob = if !isnothing(constraints)
        CorleoneDynamicOptProblem(
            layer, loss, [k => v for (k, v) in constraints]...;
            kwargs...
        )
    else
        CorleoneDynamicOptProblem(
            layer, loss;
            kwargs...
        )
    end

    # Declare the Optimization function
    ax = getaxes(u0)
    objective = let obj = prob.objective, axes = ax
        (ps, st) -> obj(ComponentArray(ps, axes), st)
    end
    if !isnothing(prob.constraints) 
        constraints = let cons = prob.constraints, axes = ax
            (res, ps, st) -> cons(res, ComponentArray(ps, axes), st)
        end
    else
        constraints = prob.constraints
    end
    opt_f = OptimizationFunction(objective, AD; cons = constraints)

    # Return the optimization problem
    return OptimizationProblem(
        opt_f, u0[:], st, lb = lb[:], ub = ub[:],
        lcons = prob.lcons, ucons = prob.ucons,
    )
end
end
