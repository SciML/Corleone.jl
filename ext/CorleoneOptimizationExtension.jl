module CorleoneOptimizationExtension
using Corleone
using Optimization
using SymbolicIndexingInterface
using ComponentArrays
using LuxCore
using Random

function Optimization.OptimizationProblem(
        layer::SingleShootingLayer,
        loss::Union{Symbol, Expr};
        AD::Optimization.ADTypes.AbstractADType = AutoForwardDiff(),
        u0::ComponentVector = ComponentArray(first(LuxCore.setup(Random.default_rng(), layer))),
        constraints::Union{Nothing, <:Dict{<:Union{Expr, Symbol}, <:NamedTuple{(:t, :bounds)}}} = nothing,
        variable_type::Type{T} = Float64,
        kwargs...
    ) where {T}

    u0 = T.(u0)

    # Our objective function
    ps, st = LuxCore.setup(Random.default_rng(), layer)
    sol, _ = layer(nothing, ps, st)
    getter = SymbolicIndexingInterface.getsym(sol, loss)

    objective = let layer = layer, ax = getaxes(ComponentArray(ps)), getter = getter
        (p, st) -> begin
            ps = ComponentArray(p, ax)
            sols, _ = layer(nothing, ps, st)
            last(getter(sols))
        end
    end

    # Bounds based on the variables
    lb, ub = Corleone.get_bounds(layer) .|> ComponentArray

    @assert all(lb .<= u0 .<= ub) "The initial variables are not within the bounds. Please check the input!"

    # Constraints
    cons = begin
        if !isnothing(constraints)

            getter_constraints = []
            for (k, v) in constraints
                push!(getter_constraints, getsym(sol, k))
            end

            sampling_cons = let layer = layer, ax = getaxes(ComponentArray(ps)), getter = getter_constraints, constraints = constraints
                (res, p, st) -> begin
                    ps = ComponentArray(p, ax)
                    sols, _ = layer(nothing, ps, st)

                    cons = map(enumerate(constraints)) do (i, (k, v))
                        # Caution: timepoints for controls need to be in sols.t!
                        idxs = map(ti -> findfirst(x -> x .== ti, sols.t), v.t)
                        getter[i](sols)[idxs]
                    end

                    res .= reduce(vcat, cons)
                end
            end

            sampling_cons
        else
            nothing
        end
    end

    lcons, ucons = begin
        if isnothing(constraints)
            nothing, nothing
        else
            _lb = reduce(
                vcat, map(enumerate(constraints)) do (i, (k, v))
                    first(v.bounds)
                end
            )
            _ub = reduce(
                vcat, map(enumerate(constraints)) do (i, (k, v))
                    first(v.bounds)
                end
            )
            _lb, _ub
        end
    end

    # Declare the Optimization function
    opt_f = OptimizationFunction(objective, AD; cons = cons)

    # Return the optimization problem
    return OptimizationProblem(
        opt_f, u0[:], st, lb = lb[:], ub = ub[:],
        lcons = lcons, ucons = ucons,
    )
end


function Optimization.OptimizationProblem(
        layer::MultipleShootingLayer,
        loss::Union{Symbol, Expr};
        AD::Optimization.ADTypes.AbstractADType = AutoForwardDiff(),
        u0::ComponentVector = ComponentArray(first(LuxCore.setup(Random.default_rng(), layer))),
        constraints = nothing, variable_type::Type{T} = Float64,
        kwargs...
    ) where {T}

    u0 = T.(u0)

    # Our objective function
    ps, st = LuxCore.setup(Random.default_rng(), layer)
    sol, _ = layer(nothing, ps, st)
    getter = SymbolicIndexingInterface.getsym(sol, loss)

    objective = let layer = layer, ax = getaxes(ComponentArray(ps))
        (p, st) -> begin
            ps = ComponentArray(p, ax)
            sols, _ = layer(nothing, ps, st)
            last(getter(sols))
        end
    end

    # Bounds based on the variables
    lb, ub = Corleone.get_bounds(layer) .|> ComponentArray

    @assert all(lb .<= u0 .<= ub) "The initial variables are not within the bounds. Please check the input!"

    nshooting = Corleone.get_number_of_shooting_constraints(layer)

    cons = begin
        if isnothing(constraints)
            shooting_constraints = let layer = layer, ax = getaxes(ComponentArray(ps))
                (res, p, st) -> begin
                    ps = ComponentArray(p, ax)
                    sols, _ = layer(nothing, ps, st)
                    Corleone.shooting_constraints!(res, sols)
                end
            end
            shooting_constraints
        else
            getter_constraints = []
            for (k, v) in constraints
                push!(getter_constraints, getsym(sol, k))
            end

            shooting_constraints = let layer = layer, ax = getaxes(ComponentArray(ps)), getter = getter_constraints, constraints = constraints
                (res, p, st) -> begin
                    ps = ComponentArray(p, ax)
                    sols, _ = layer(nothing, ps, st)
                    matching = Corleone.shooting_constraints(sols)

                    cons = map(enumerate(constraints)) do (i, (k, v))
                        # Caution: timepoints for controls need to be in sols.t!
                        idxs = map(ti -> findfirst(x -> x .== ti, sols.t), v.t)
                        getter[i](sols)[idxs]
                    end

                    res .= vcat(reduce(vcat, cons), matching)
                end
            end

            shooting_constraints
        end
    end

    lcons, ucons = begin
        if !isnothing(constraints)
            _lb = reduce(
                vcat, map(enumerate(constraints)) do (i, (k, v))
                    first(v.bounds)
                end
            )
            _ub = reduce(
                vcat, map(enumerate(constraints)) do (i, (k, v))
                    first(v.bounds)
                end
            )
            vcat(_lb, zeros(T, nshooting)), vcat(_ub, zeros(T, nshooting))
        else
            zeros(T, nshooting), zeros(T, nshooting)
        end
    end

    # Declare the Optimization function
    opt_f = OptimizationFunction(
        objective, AD;
        cons = shooting_constraints
    )

    # Return the optimization problem
    return OptimizationProblem(
        opt_f, u0[:], st, lb = lb[:], ub = ub[:],
        lcons = lcons, ucons = ucons,
    )
end
end
