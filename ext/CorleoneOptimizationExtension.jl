module CorleoneOptimizationExtension
using Corleone
using Optimization
using SymbolicIndexingInterface
using ComponentArrays
using LuxCore
using Random
@info "Loading CorleoneOptimizationExtension..."

function Optimization.OptimizationProblem(layer::SingleShootingLayer,
        loss::Union{Symbol,Expr};
        AD::Optimization.ADTypes.AbstractADType = AutoForwardDiff(),
        u0::ComponentVector = ComponentArray(first(LuxCore.setup(Random.default_rng(), layer))),
        p = SciMLBase.NullParameters(),
        integer_constraints::Bool = false,
        constraints::Union{Nothing, <:Dict{<:Union{Expr,Symbol},<:NamedTuple{(:t,:bounds)}}} = nothing,
        variable_type::Type{T} = Float64,
        kwargs...) where {T}

    u0 = T.(u0)
    p = !isa(p, SciMLBase.NullParameters) ? T.(p) : p

    # Our objective function
    ps, st = LuxCore.setup(Random.default_rng(), layer)
    sol, _ = layer(nothing, ps, st)
    getter = SymbolicIndexingInterface.getsym(sol, loss)

    objective = let layer = layer, st = st, ax = getaxes(ComponentArray(ps)), getter=getter
        (p, ::Any) -> begin
            ps = ComponentArray(p, ax)
            sols, _ = layer(nothing, ps, st)
            last(getter(sols))
        end
    end

    # Bounds based on the variables
    lb, ub = Corleone.get_bounds(layer) .|> ComponentArray

    @assert all(lb .<= u0 .<= ub) "The initial variables are not within the bounds. Please check the input!"

    # No integers
    integrality = Bool.(u0 * 0)

    # Constraints
    cons = begin
        if !isnothing(constraints)

        getter_constraints = []
        for (k,v) in constraints
            push!(getter_constraints, getsym(sol, k))
        end

        sampling_cons = let layer = layer, st = st, ax = getaxes(ComponentArray(ps)), getter=getter_constraints, constraints=constraints
            (res, p, ::Any) -> begin
                ps = ComponentArray(p, ax)
                sols, _ = layer(nothing, ps, st)

                cons = map(enumerate(constraints)) do (i, (k,v))
                    idxs = map(ti -> findfirst(x -> x .== ti , sols.t), v.t)
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
            _lb = reduce(vcat, map(enumerate(constraints)) do (i, (k,v))
                first(v.bounds)
            end)
            _ub = reduce(vcat, map(enumerate(constraints)) do (i, (k,v))
                first(v.bounds)
            end)
            _lb, _ub
        end
    end

    # Declare the Optimization function
    opt_f = OptimizationFunction(objective, AD; cons=cons)

    # Return the optimization problem
    OptimizationProblem(opt_f, u0[:], p, lb = lb[:], ub = ub[:], int = integrality[:],
        lcons = lcons, ucons = ucons,
    )
end


function Optimization.OptimizationProblem(layer::MultipleShootingLayer,
        loss::Union{Symbol,Expr};
        AD::Optimization.ADTypes.AbstractADType = AutoForwardDiff(),
        u0::ComponentVector = ComponentArray(first(LuxCore.setup(Random.default_rng(), layer))), p = SciMLBase.NullParameters(),
        integer_constraints::Bool = false,
        constraints = nothing, variable_type::Type{T} = Float64,
        kwargs...) where {T}

    u0 = T.(u0)
    p = !isa(p, SciMLBase.NullParameters) ? T.(p) : p

    # Our objective function
    ps, st = LuxCore.setup(Random.default_rng(), layer)
    sols, _ = layer(nothing, ps, st)
    getter = SymbolicIndexingInterface.getsym(sols, loss)

    objective = let layer = layer, st = st, ax = getaxes(ComponentArray(ps))
        (p, ::Any) -> begin
            ps = ComponentArray(p, ax)
            sols, _ = layer(nothing, ps, st)
            last(getter(sols))
        end
    end

    # Bounds based on the variables
    lb, ub = Corleone.get_bounds(layer) .|> ComponentArray

    @assert all(lb .<= u0 .<= ub) "The initial variables are not within the bounds. Please check the input!"

    # No integers
    integrality = Bool.(u0 * 0)

    nshooting = Corleone.get_number_of_shooting_constraints(layer)

    shooting_constraints = let layer = layer, st = st, ax = getaxes(ComponentArray(ps))
        (res, p, ::Any) -> begin
            ps = ComponentArray(p, ax)
            sols, _ = layer(nothing, ps, st)
            Corleone.shooting_constraints!(res, sols)
        end
    end

    # Declare the Optimization function
    opt_f = OptimizationFunction(objective, AD;
        cons = shooting_constraints)

    # Return the optimization problem
    OptimizationProblem(opt_f, u0[:], p, lb = lb[:], ub = ub[:], int = integrality[:],
        lcons = zeros(T, nshooting), ucons = zeros(T, nshooting),
    )
end


end
