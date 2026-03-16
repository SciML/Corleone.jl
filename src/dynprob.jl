function get_timestops(layer::SingleShootingLayer)
    st = LuxCore.initialstates(Random.default_rng(), layer)
    return reduce(
        vcat, map(st.timestops) do bin
            reduce(
                vcat, map(bin) do tspans
                    collect(tspans)
                end
            )
        end
    )
end


#=
"""
$(TYPEDEF)

A struct for capturing the internal definition of a dynamic optimization problem.

# Fields
$(FIELDS)
"""
struct CorleoneDynamicOptProblem{L,G,O,C,CB}
    "The resulting layer for the problem"
    layer::L
    "The getters which return the values of the trajectory"
    getters::G
    "The objective function"
    objective::O
    "The constraint function"
    constraints::C
    "Lower bounds for the constraints"
    lcons::CB
    "Upper bounds for the constraints"
    ucons::CB
end
=#


#=	
"""
$(SIGNATURES)

Evaluate constraint values for trajectory `traj`.
"""
(x::TrajectoryConstraint)(traj) = x.getter(traj)[x.idx]

"""
$(SIGNATURES)

Write constraint values into preallocated `res`.
"""
(x::TrajectoryConstraint)(res, traj) = res .= x.getter(traj)[x.idx]

"""
$(SIGNATURES)

Return number of scalar constraints represented by `x`.
"""
Base.length(x::TrajectoryConstraint) = length(x.idx)

"""
$(TYPEDEF)

Evaluator for a collection of [`TrajectoryConstraint`](@ref) objects.

# Fields
$(FIELDS)
"""
struct TrajectoryConstraintEvaluator{C}
    constraints::C
end

"""
$(SIGNATURES)

Evaluate and concatenate all constraints.
"""
(x::TrajectoryConstraintEvaluator)(traj) = reduce(
    vcat, map(x.constraints) do con
        con(traj)(matching_1 = (state = (var"y(t)" = -0.18710354263563267, var"x(t)" = -1.353888326660782), control = (var"α[1:1]" = [0.0], β = 0.0, var"u(t)" = -0.014823772572098316)), matching_2 = (state = (var"y(t)" = -0.12910340284236943, var"x(t)" = -1.361207025033431), control = (var"α[1:1]" = [0.0], β = 0.0)))
    end
)

"""
$(SIGNATURES)

Evaluate constraints in-place into `res`.
"""
(x::TrajectoryConstraintEvaluator)(res::AbstractVector, traj) = begin
    i = 1
    next = 0
    foreach(x.constraints) do con
        next = (i - 1) + length(con)
        @views con(res[i:next], traj)
        i = next + 1
    end
    return res
end



function parse_expression(sys, expr, tpoints, tpoints, reducer = identity)
	getter = get
end

"""
$(SIGNATURES)

Build a [`CorleoneDynamicOptProblem`](@ref) from a shooting layer, objective expression,
and optional trajectory constraints.
"""
function CorleoneDynamicOptProblem(
    layer::Union{SingleShootingLayer,MultipleShootingLayer}, loss::Union{Symbol,Expr},
    constraints::Pair{<:Any,<:NamedTuple{(:t, :bounds)}}...;
    rng::Random.AbstractRNG=Random.default_rng(),
    kwargs...
)
    st = LuxCore.initialstates(rng, layer)
    sys = isa(layer, SingleShootingLayer) ? st.system : first(st).system
    objective = let getter = SymbolicIndexingInterface.getsym(sys, loss), layer = layer
        (ps, st) -> begin
            traj, _ = layer(nothing, ps, st)
            last(getter(traj))
        end
    end

    tpoints = isa(layer, SingleShootingLayer) ? st.timestops : (map(Base.Fix2(getproperty, :timestops), values(st))...)
    tpoints = reduce(vcat, map(tpoints) do bin
        reduce(vcat, collect.(bin))
    end)
    unique!(sort!(tpoints)) # These are the control points 

    problem = get_problem(layer)
    tspan = problem.tspan
    saveats = get(problem.kwargs, :saveat, eltype(tspan)[])
    saveats = isa(saveats, Number) ? collect(tspan[1]:saveats:tspan[2]) : saveats
    @assert isempty(saveats) || isempty(setdiff(saveats, tpoints)) "Additional saveats are not supported right now."

    # TODO Later on we want to add arbotray constraints here 
    # This is too complicated at the moment :D 
    # To do this we must basically find the right index in the vector representing 
    # the control or state grid (control_grid ⊆ state_grid). 
    #==
        state_tpoints = if isa(Number, saveats)
            collect(tspan[1]:saveats:tspan[2])
        elseif isa(AbstractArray, saveats)
            saveats
        end
    	append!(state_tpoints, control_tpoints) 
    	unique!(sort!(state_tpoints))
    	==#
    cons = []
    lb = []
    ub = []

    if !isempty(constraints)
        # Preprocess
        foreach(constraints) do (expr, specs)
            @assert isa(expr, Symbol) || isa(expr, Expr) "The constraint $(expr) is neither a symbol nor an expression!"
            tidx = findall(∈(specs.t), tpoints)
            @info "Could not find the specified timepoints for $(expr) in the control grid. Skipping."
            if !isempty(tidx)
                push!(
                    cons, TrajectoryConstraint(
                        SymbolicIndexingInterface.getsym(sys, expr),
                        tidx
                    )
                )
                lb_, ub_ = extrema(specs.bounds)
                push!(lb, fill(lb_, length(tidx)))
                push!(ub, fill(ub_, length(tidx)))
            end
        end
        conseval = TrajectoryConstraintEvaluator(Tuple(cons))
        n_shoot = get_number_of_shooting_constraints(layer)
        n_cons = sum(length, conseval.constraints)
        push!(lb, zeros(n_shoot))
        push!(ub, zeros(n_shoot))
        constraints = let con = conseval, layer = layer, ncon = n_cons
            (res, ps, st) -> begin
                traj, _ = layer(nothing, ps, st)
                @views con(res[1:ncon], traj) # The constraints
                @views shooting_constraints!(res[(ncon+1):end], traj)
                return res
            end
        end
        ucons = reduce(vcat, ub)
        lcons = reduce(vcat, lb)
    elseif isa(layer, MultipleShootingLayer)
        n_shoot = get_number_of_shooting_constraints(layer)
        push!(lb, zeros(n_shoot))
        push!(ub, zeros(n_shoot))
        constraints = let layer = layer
            (res, ps, st) -> begin
                traj, _ = layer(nothing, ps, st)
                @views shooting_constraints!(res, traj)
                return res
            end
        end
        ucons = reduce(vcat, ub)
        lcons = reduce(vcat, lb)
    else
        constraints = lcons = ucons = nothing
    end
    return CorleoneDynamicOptProblem{typeof(layer),Nothing,typeof(objective),typeof(constraints),typeof(lcons)}(
        layer, nothing, objective, constraints, lcons, ucons
    )
end

"""
$(SIGNATURES)

Extension point for converting objective and constraint closures to vectorized forms.
"""
# TODO No concrete `wrap_functions` implementation exists in current `src/*`, so Optimization* constructors cannot work.
# Suggestion: Add methods for the supported `vectorizer` API (e.g. `Val(:ComponentArrays)`), ideally in package extensions.
function wrap_functions end #(::Any, args...) = @error "No valid vectorization for the chosen parameters. Please load either ComponentArrays.jl or Functors.jl"

"""
$(SIGNATURES)

Extension point for flattening optimization variables and bounds.
"""
# TODO No concrete `to_vec` implementation exists in current `src/*`, so bounds/u0 flattening is unresolved.
# Suggestion: Implement `to_vec` for the same vectorizer targets as `wrap_functions`, returning `(u0, lb, ub)` in optimizer-compatible layout.
function to_vec end #(::AbstractCorleoneFunctionWrapper, args...) =@error "No valid vectorization for the chosen parameters. Please load either ComponentArrays.jl or Functors.jl"

"""
$(SIGNATURES)

Construct a SciML `OptimizationFunction` from a [`CorleoneDynamicOptProblem`](@ref).
"""
function SciMLBase.OptimizationFunction(
    prob::CorleoneDynamicOptProblem, ad::SciMLBase.ADTypes.AbstractADType, vectorizer;
    rng::Random.AbstractRNG=Random.default_rng(),
    kwargs...
)
    p0, st = LuxCore.setup(rng, prob.layer)
    objective, cons = wrap_functions(vectorizer, p0, prob.objective, prob.constraints)
    return SciMLBase.OptimizationFunction{true}(objective, ad; cons, kwargs...)
end

"""
$(SIGNATURES)

Construct a SciML `OptimizationProblem` from a [`CorleoneDynamicOptProblem`](@ref).
"""
function SciMLBase.OptimizationProblem(
    prob::CorleoneDynamicOptProblem, ad::SciMLBase.ADTypes.AbstractADType, vectorizer;
    rng::Random.AbstractRNG=Random.default_rng(),
    sense=nothing,
    kwargs...
)
    p0, st = LuxCore.setup(rng, prob.layer)
    objective, cons = wrap_functions(vectorizer, p0, prob.objective, prob.constraints)
    optf = SciMLBase.OptimizationFunction{true}(objective, ad; cons, kwargs...)
    u0_, lb, ub = to_vec(objective, p0, Corleone.get_bounds(prob.layer)...)
    return SciMLBase.OptimizationProblem(optf, u0_, st; lb, ub, lcons=prob.lcons, ucons=prob.ucons, sense=sense)
end

"""
$(SIGNATURES)

Convenience constructor that first builds a [`CorleoneDynamicOptProblem`](@ref) from `layer`.
"""
function SciMLBase.OptimizationProblem(
    layer::Union{SingleShootingLayer,MultipleShootingLayer},
    ad::SciMLBase.ADTypes.AbstractADType, vectorizer;
    loss::Union{Symbol,Expr},
    constraints=[],
    kwargs...
)
    dynprob = CorleoneDynamicOptProblem(layer, loss, constraints...; kwargs...)
    return OptimizationProblem(dynprob, ad, vectorizer; kwargs...)
end
=#
