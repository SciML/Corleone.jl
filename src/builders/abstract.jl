function Base.show(io::IO, builder::AbstractBuilder)
    (; system) = builder
    cost = ModelingToolkit.get_consolidate(system)(ModelingToolkit.get_costs(system))
    cons = ModelingToolkit.constraints(system)
    eqs = ModelingToolkit.equations(system)
    obs = ModelingToolkit.observed(system)
    println(io, "min $(cost)")
    println(io, "")
    println(io, "s.t.")
    println(io, "")
    if !isempty(eqs)
        println(io, "Dynamics $(nameof(system))")
        foreach(eqs) do eq
            println(io, "     ", eq)
        end
    end
    if !isempty(obs)
        println(io, "Observed $(nameof(system))")
        foreach(ModelingToolkit.observed(system)) do eq
            println(io, "     ", eq)
        end
    end
    if !isempty(cons)
        println(io, "Constraints")
        foreach(cons) do eq
            println(io, "     ", eq)
        end
    end
end

__process_costs(::AbstractBuilder) = true 
__process_constraints(::AbstractBuilder) = true

ModelingToolkit.iscomplete(builder::AbstractBuilder) = ModelingToolkit.iscomplete(builder.system)

function SciMLBase.OptimizationFunction{IIP}(builder::AbstractBuilder, adtype::SciMLBase.ADTypes.AbstractADType, alg::DEAlgorithm, args...; kwargs...) where {IIP}
    if !ModelingToolkit.iscomplete(builder)
        @debug "Processing the optimization problem..."
        builder = builder()
    end

    (; system) = builder

    constraints = ModelingToolkit.constraints(system)
    costs = ModelingToolkit.get_costs(system)
    consolidate = ModelingToolkit.get_consolidate(system)
    objective = OptimalControlFunction{true}(
        costs, builder, alg, args...; consolidate=consolidate, kwargs...
    )
    if !isempty(constraints)
        constraints = OptimalControlFunction{IIP}(
            map(x -> x.lhs, constraints), builder, alg, args...; kwargs...
        )
    else
        constraints = nothing
    end
    return OptimizationFunction{IIP}(
        objective, adtype, cons=constraints,
    )
end

function SciMLBase.OptimizationProblem{IIP}(builder::AbstractBuilder, adtype::SciMLBase.ADTypes.AbstractADType, alg::DEAlgorithm, args...; initialization::AbstractNodeInitialization = DefaultsInitialization(), kwargs...) where {IIP}
    if !ModelingToolkit.iscomplete(builder)
        @debug "Processing the optimization problem..."
        builder = builder()
    end
    (; system) = builder 
    constraints = ModelingToolkit.constraints(system)
    f = OptimizationFunction{IIP}(builder, adtype, alg, args...; kwargs...)
    predictor = initialization(f.f.predictor)
    u0 = get_p0(predictor)
    lb, ub = get_bounds(predictor)
    if !isempty(constraints)
        lcons = zeros(eltype(u0), size(constraints, 1))
        ucons = zeros(eltype(u0), size(constraints, 1))
        for (i, c) in enumerate(constraints)
            isa(c, Equation) && continue
            lcons[i] = eltype(u0)(-Inf)
        end
    else
        lcons = ucons = nothing
    end
    OptimizationProblem{IIP}(f, u0; lb, ub, lcons, ucons)
end
