module CorleoneOEDOptimizationExtension
using CorleoneOED
using Corleone
using ComponentArrays
using LuxCore
using Optimization
using SymbolicIndexingInterface
using Random

n_observed(layer::OEDLayer) = length(layer.sampling_indices)
n_observed(layer::MultiExperimentLayer{<:Any, <:Any, false}) = layer.n_exp * length(layer.layers.sampling_indices)
n_observed(layer::MultiExperimentLayer{<:Any, <:Any, true}) = sum(map(x->length(x.sampling_indices), layer.layers))

Corleone.get_number_of_shooting_constraints(oed::OEDLayer{<:Any, <:Any, <:Any, <:MultipleShootingLayer}) = Corleone.get_number_of_shooting_constraints(oed.layer)
Corleone.get_number_of_shooting_constraints(oed::OEDLayer{<:Any, <:Any, <:Any, <:SingleShootingLayer}) = 0
Corleone.get_number_of_shooting_constraints(multi::MultiExperimentLayer{<:Any, <:Any, false, <:MultipleShootingLayer}) = multi.n_exp * Corleone.get_number_of_shooting_constraints(multi.layers)
Corleone.get_number_of_shooting_constraints(multi::MultiExperimentLayer{<:Any, <:Any, true, <:MultipleShootingLayer}) = sum(map(Corleone.get_number_of_shooting_constraints, multi.layers))
Corleone.get_number_of_shooting_constraints(multi::MultiExperimentLayer{<:Any, <:Any, <:Any, <:SingleShootingLayer}) = 0

default_M(layer::OEDLayer{false}) = zeros(n_observed(layer)) .+ last(Corleone.get_tspan(layer.layer))
default_M(layer::OEDLayer{true}) = zeros(n_observed(layer)) .+ [length(x.t) for x in layer.controls]
default_M(layer::MultiExperimentLayer{false, <:Any, false}) = reduce(vcat, [default_M(layer.layers) for _ in 1:layer.n_exp])
default_M(layer::MultiExperimentLayer{false, <:Any, true}) = reduce(vcat, map(default_M, layer.layers))

function extract_constraint_bounds(layer::Union{OEDLayer,MultiExperimentLayer}, constraints::Nothing, M)
    nshooting = Corleone.get_number_of_shooting_constraints(layer)
    lcons, ucons = begin
        zeros(nshooting+length(M)), vcat(zeros(nshooting),M)
    end
    return lcons, ucons
end

function extract_constraint_bounds(layer::OEDLayer, constraints::Dict, M)
    nshooting = Corleone.get_number_of_shooting_constraints(layer)
    lcons, ucons = begin
        _lb = reduce(vcat, map(enumerate(constraints)) do (i, (k,v))
            first(v.bounds)
        end)
        _ub = reduce(vcat, map(enumerate(constraints)) do (i, (k,v))
            last(v.bounds)
        end)
        vcat(_lb, zeros(nshooting+length(M))), vcat(_ub, zeros(nshooting), M)
    end
    return lcons, ucons
end

function extract_constraint_bounds(layer::MultiExperimentLayer, constraints::NamedTuple{fields}, M) where {fields}
    nshooting = Corleone.get_number_of_shooting_constraints(layer)
    lcons, ucons = begin
        if isnothing(constraints)
            zeros(nshooting+length(M)), vcat(zeros(nshooting),M)
        else
            _lb = reduce(vcat, map(fields) do field
                local_constraint = getproperty(constraints, field)
                reduce(vcat, map(enumerate(local_constraint)) do (i, (k,v))
                    first(v.bounds)
                end)
            end)
            _ub = reduce(vcat, map(fields) do field
                local_constraint = getproperty(constraints, field)
                reduce(vcat, map(enumerate(local_constraint)) do (i, (k,v))
                    last(v.bounds)
                end)
            end)
            vcat(_lb, zeros(nshooting), zero(M)), vcat(_ub, zeros(nshooting), M)
        end
    end
    return lcons, ucons
end

function setup_constraints(layer::OEDLayer{<:Any, true, <:Any, <:SingleShootingLayer}, sol, constraints::Nothing)
    ps, st = LuxCore.setup(Random.default_rng(), layer)
    sampling_cons = let layer = layer, ax = getaxes(ComponentArray(ps))
        (res, p, st) -> begin
            ps = ComponentArray(p, ax)
            CorleoneOED.get_sampling_sums!(res, layer, nothing, ps, st)
        end
    end
    return sampling_cons
end

function setup_constraints(layer::OEDLayer{<:Any, true, <:Any, <:SingleShootingLayer}, sol, constraints)
    ps, st = LuxCore.setup(Random.default_rng(), layer)
    getter_constraints = []
    for (k,v) in constraints
        push!(getter_constraints, getsym(sol, k))
    end

    sampling_cons = let layer = layer, ax = getaxes(ComponentArray(ps)), getter=getter_constraints, constraints=constraints
        (res, p, st) -> begin
            ps = ComponentArray(p, ax)
            sols, _ = layer(nothing, ps, st)
            sampling = CorleoneOED.get_sampling_sums(layer, nothing, ps, st)

            cons = map(enumerate(constraints)) do (i, (k,v))
                # Caution: timepoints for controls need to be in sols.t!
                idxs = map(ti -> findfirst(x -> x .== ti , sols.t), v.t)
                getter[i](sols)[idxs]
            end

            res .= vcat(reduce(vcat, cons), sampling)
        end
    end

    return sampling_cons
end

function setup_constraints(layer::OEDLayer{<:Any, true, false, <:MultipleShootingLayer}, sol, constraints::Nothing)
    ps, st = LuxCore.setup(Random.default_rng(), layer)
    sampling_cons = let layer = layer, ax = getaxes(ComponentArray(ps))
        (res, p, st) -> begin
            ps = ComponentArray(p, ax)
            sols, _ = layer(nothing, ps, st)
            shooting = Corleone.shooting_constraints(sols)
            sampling = CorleoneOED.get_sampling_sums(layer, nothing, ps, st)
            res .= vcat(shooting, sampling)
        end
    end
    return sampling_cons
end

function setup_constraints(layer::OEDLayer{<:Any, true, false, <:MultipleShootingLayer}, sol, constraints)
    ps, st = LuxCore.setup(Random.default_rng(), layer)
    getter_constraints = []
    for (k,v) in constraints
        push!(getter_constraints, getsym(sol, k))
    end

    sampling_cons = let layer = layer, ax = getaxes(ComponentArray(ps)), getter=getter_constraints, constraints=constraints
        (res, p, st) -> begin
            ps = ComponentArray(p, ax)
            sols, _ = layer(nothing, ps, st)
            sampling = CorleoneOED.get_sampling_sums(layer, nothing, ps, st)
            shooting = Corleone.shooting_constraints(sols)
            cons = map(enumerate(constraints)) do (i, (k,v))
                # Caution: timepoints for controls need to be in sols.t!
                idxs = map(ti -> findfirst(x -> x .== ti , sols.t), v.t)
                getter[i](sols)[idxs]
            end

            res .= vcat(reduce(vcat, cons), shooting, sampling)
        end
    end

    return sampling_cons
end

function setup_constraints(layer::MultiExperimentLayer{<:Any, <:Any, <:Any, <:SingleShootingLayer}, sols, constraints::Nothing)
    ps, st = LuxCore.setup(Random.default_rng(), layer)
    sampling_cons = let layer = layer, ax = getaxes(ComponentArray(ps))
        (res, p, st) -> begin
            ps = ComponentArray(p, ax)
            CorleoneOED.get_sampling_sums!(res, layer, nothing, ps, st)
        end
    end
    return sampling_cons
end

function setup_constraints(layer::MultiExperimentLayer{<:Any, <:Any, <:Any, <:SingleShootingLayer}, sols, constraints::NamedTuple{fields}) where {fields}
    ps, st = LuxCore.setup(Random.default_rng(), layer)
    getter_constraints = []
    for (i,field) in enumerate(fields)
        local_constraints = getproperty(constraints, field)
        push!(getter_constraints, map(local_constraints) do (k,v)
            getsym(sols[i], k)
        end)
    end

    sampling_cons = let layer = layer, ax = getaxes(ComponentArray(ps)), fields=fields, getter=getter_constraints, constraints=constraints
        (res, p, st) -> begin
            ps = ComponentArray(p, ax)
            sols, _ = layer(nothing, ps, st)
            sampling = CorleoneOED.get_sampling_sums(layer, nothing, ps, st)
            cons = map(enumerate(fields)) do (i,field)
                local_constraints = getproperty(constraints, field)
                reduce(vcat, map(zip(local_constraints,getter[i])) do ((k,v), getter_i)
                    # Caution: timepoints for controls need to be in sols.t!
                    idxs = map(ti -> findfirst(x -> x .== ti , sols.t), v.t)
                    getter_i(sols[i])[idxs]
                end)
            end

            res .= vcat(reduce(vcat, cons), sampling)
        end
    end

    return sampling_cons
end

function setup_constraints(layer::MultiExperimentLayer{<:Any, <:Any, <:Any, <:MultipleShootingLayer}, sols, constraints::Nothing)
    ps, st = LuxCore.setup(Random.default_rng(), layer)
    sampling_cons = let layer = layer, ax = getaxes(ComponentArray(ps))
        (res, p, st) -> begin
            ps = ComponentArray(p, ax)
            shooting = Corleone.shooting_constraints(sols)
            matching = CorleoneOED.get_sampling_sums(layer, nothing, ps, st)
            res .= vcat(shooting, matching)
        end
    end
    return sampling_cons
end

function setup_constraints(layer::MultiExperimentLayer{<:Any, <:Any, <:Any, <:MultipleShootingLayer}, sols, constraints::NamedTuple{fields}) where {fields}
    ps, st = LuxCore.setup(Random.default_rng(), layer)
    getter_constraints = []
    for (i,field) in enumerate(fields)
        local_constraints = getproperty(constraints, field)
        push!(getter_constraints, map(local_constraints) do (k,v)
            getsym(sols[i], k)
        end)
    end

    sampling_cons = let layer = layer, ax = getaxes(ComponentArray(ps)), fields=fields, getter=getter_constraints, constraints=constraints
        (res, p, st) -> begin
            ps = ComponentArray(p, ax)
            sols, _ = layer(nothing, ps, st)
            sampling = CorleoneOED.get_sampling_sums(layer, nothing, ps, st)
            shooting = Corleone.shooting_constraints(sols)
            cons = map(enumerate(fields)) do (i,field)
                local_constraints = getproperty(constraints, field)
                reduce(vcat, map(zip(local_constraints,getter[i])) do ((k,v), getter_i)
                    # Caution: timepoints for controls need to be in sols.t!
                    idxs = map(ti -> findfirst(x -> x .== ti , sols.t), v.t)
                    getter_i(sols[i])[idxs]
                end)
            end

            res .= vcat(reduce(vcat, cons), shooting, sampling)
        end
    end

    return sampling_cons
end

function Optimization.OptimizationProblem(layer::Union{OEDLayer{<:Any, true, false}, MultiExperimentLayer{<:Any, false}},
        crit::CorleoneOED.AbstractCriterion;
        AD::Optimization.ADTypes.AbstractADType = AutoForwardDiff(),
        u0::ComponentVector = ComponentArray(first(LuxCore.setup(Random.default_rng(), layer))),
        integer_constraints::Bool = false,
        constraints::Union{Nothing, <:Dict{<:Union{Expr,Symbol},<:NamedTuple{(:t,:bounds)}}} = nothing,
        variable_type::Type{T} = Float64,
        M = default_M(layer),
        kwargs...) where {T}

    u0 = T.(u0)

    # Our objective function
    ps, st = LuxCore.setup(Random.default_rng(), layer)
    sol, _ = layer(nothing, ps, st)
    objective = let layer = layer, ax = getaxes(ComponentArray(ps))
        (p, st) -> begin
            ps = ComponentArray(p, ax)
            first(crit(layer, nothing, ps, st))
        end
    end

    @assert length(M) == n_observed(layer) "Dimensions of upper bound on sampling constraints do not match, expected $(n_observed(layer)), got $(length(M))."

    # Bounds based on the variables
    lb, ub = Corleone.get_bounds(layer) .|> ComponentArray

    @assert all(lb .<= u0 .<= ub) "The initial variables are not within the bounds. Please check the input!"

    # No integers
    integrality = Bool.(u0 * 0)

    # Constraints
    cons = setup_constraints(layer, sol, constraints)
    lcons, ucons = extract_constraint_bounds(layer, constraints, M)

    # Declare the Optimization function
    opt_f = OptimizationFunction(objective, AD; cons=cons)

    # Return the optimization problem
    OptimizationProblem(opt_f, u0[:], st, lb = lb[:], ub = ub[:], int = integrality[:],
        lcons = lcons, ucons = ucons,
    )
end

function Optimization.OptimizationProblem(layer::Union{OEDLayer{<:Any, true, true}, MultiExperimentLayer{<:Any,true}},
        crit::CorleoneOED.AbstractCriterion;
        AD::Optimization.ADTypes.AbstractADType = AutoForwardDiff(),
        u0::ComponentVector = ComponentArray(first(LuxCore.setup(Random.default_rng(), layer))),
        integer_constraints::Bool = false,
        constraints::Union{Nothing, <:Dict{<:Union{Expr,Symbol},<:NamedTuple{(:t,:bounds)}}} = nothing,
        variable_type::Type{T} = Float64,
        M = default_M(layer),
        kwargs...) where {T}

    u0 = T.(u0)

    # Our objective function
    ps, st = LuxCore.setup(Random.default_rng(), layer)
    p = ComponentArray(ps)
    if isa(layer, OEDLayer)
        p.controls .= 1.0 # Solve initial system with all ones for sampling
    else # MultiExperiments
        for field in keys(st)
            p[field].controls .= 1.0
        end
    end

    sol, _ = layer(nothing, p, st)

    objective = let layer = layer, ax = getaxes(ComponentArray(ps)), sol=sol
        (p, st) -> begin
            ps = ComponentArray(p, ax)
            first(crit(CorleoneOED.__fisher_information(layer, sol, ps, st)))
        end
    end

    @assert length(M) == n_observed(layer) "Dimensions of upper bound on sampling constraints do not match, expected $(n_observed(layer)), got $(length(M))."

    # Bounds based on the variables
    lb, ub = Corleone.get_bounds(layer) .|> ComponentArray

    @assert all(lb .<= u0 .<= ub) "The initial variables are not within the bounds. Please check the input!"

    # No integers
    integrality = Bool.(u0 * 0)

    # Constraints
    cons = setup_constraints(layer, sol, constraints)
    lcons, ucons = extract_constraint_bounds(layer, constraints, M)

    # Declare the Optimization function
    opt_f = OptimizationFunction(objective, AD; cons=cons)

    # Return the optimization problem
    OptimizationProblem(opt_f, u0[:], st, lb = lb[:], ub = ub[:], int = integrality[:],
        lcons = lcons, ucons = ucons,
    )
end

end
