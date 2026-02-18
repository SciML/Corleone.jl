"""
$(TYPEDEF)

A struct for capturing the internal definition of a dynamic optimization problem. 

# Fields 
$(FIELDS)
"""
struct CorleoneDynamicOptProblem{L, G, O, C, CB}
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


_collect_tspans((x, _)::NTuple{2, <:Number}, rest...) = (x, _collect_tspans(rest...)...)
_collect_tspans(x::NTuple{2, <:Number}) = x
_collect_tspans(x::Tuple, rest...) = (_collect_tspans(x...)..., _collect_tspans(rest...)...)
_collect_tspans(x::Tuple) = _collect_tspans(x...)

struct TrajectoryConstraint{G, I}
    getter::G
    idx::I
end

(x::TrajectoryConstraint)(traj) = x.getter(traj)[x.idx]
(x::TrajectoryConstraint)(res, traj) = res .= x.getter(traj)[x.idx]
Base.length(x::TrajectoryConstraint) = length(x.idx)

struct TrajectoryConstraintEvaluator{C}
    constraints::C
end

(x::TrajectoryConstraintEvaluator)(traj) = reduce(
    vcat, map(x.constraints) do con
        con(traj)
    end
)

(x::TrajectoryConstraintEvaluator)(res::AbstractVector, traj) = begin
    i = 1
    next = 0
    foreach(x.constraints) do con
        next = (i - 1) + length(con)
        @views con(res[i:next], traj)
        i = next
    end
    return res
end

function CorleoneDynamicOptProblem(
        layer::Union{SingleShootingLayer, MultipleShootingLayer}, loss::Union{Symbol, Expr},
        constraints::Pair{<:Any, <:NamedTuple{(:t, :bounds)}}...;
        rng::Random.AbstractRNG = Random.default_rng(),
        kwargs...
    )
    st = LuxCore.initialstates(rng, layer)
    sys = isa(layer, SingleShootingLayer) ? st.symcache : first(st).symcache
    objective = let getter = SymbolicIndexingInterface.getsym(sys, loss), layer = layer
        (ps, st) -> begin
            traj, _ = layer(nothing, ps, st)
            last(getter(traj))
        end
    end
    # Collect all the points due to control etc
    tpoints = if isa(layer, SingleShootingLayer)
        reduce(vcat, _collect_tspans(st.tspans...))
    else
        reduce(
            vcat, map(st) do st_
                reduce(vcat, _collect_tspans(st_.tspans...))
            end
        )
    end
    cons = []
    lb = []
    ub = []
    if !isempty(constraints)
        # Preprocess
        foreach(constraints) do (expr, specs)
            @assert isa(expr, Symbol) || isa(expr, Expr) "The constraint $(expr) is neither a symbol nor an expression!"
            tidx = findall(∈(specs.t), tpoints)
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
                @views shooting_constraints!(res[(ncon + 1):end], traj)
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
    return CorleoneDynamicOptProblem{typeof(layer), Nothing, typeof(objective), typeof(constraints), typeof(lcons)}(
        layer, nothing, objective, constraints, lcons, ucons
    )
end

function wrap_functions end #(::Any, args...) = @error "No valid vectorization for the parameters choosen. Please load either ComponentArrays.jl or Functors.jl"
function to_vec end #(::AbstractCorleoneFunctionWrapper, args...) =@error "No valid vectorization for the parameters choosen. Please load either ComponentArrays.jl or Functors.jl"

function SciMLBase.OptimizationFunction(
        prob::CorleoneDynamicOptProblem, ad::SciMLBase.ADTypes.AbstractADType, vectorizer;
        rng::Random.AbstractRNG = Random.default_rng(),
        kwargs...
    )
    p0, st = LuxCore.setup(rng, prob.layer)
    objective, cons = wrap_functions(vectorizer, p0, prob.objective, prob.constraints)
    return SciMLBase.OptimizationFunction{true}(objective, ad; cons, kwargs...)
end

function SciMLBase.OptimizationProblem(
        prob::CorleoneDynamicOptProblem, ad::SciMLBase.ADTypes.AbstractADType, vectorizer;
        rng::Random.AbstractRNG = Random.default_rng(),
        sense = nothing,
        kwargs...
    )
    p0, st = LuxCore.setup(rng, prob.layer)
    objective, cons = wrap_functions(vectorizer, p0, prob.objective, prob.constraints)
    optf = SciMLBase.OptimizationFunction{true}(objective, ad; cons, kwargs...)
    u0_, lb, ub = to_vec(objective, p0, Corleone.get_bounds(prob.layer)...)
    return SciMLBase.OptimizationProblem(optf, u0_, st; lb, ub, lcons = prob.lcons, ucons = prob.ucons, sense = sense)
end

function SciMLBase.OptimizationProblem(
        layer::Union{SingleShootingLayer, MultipleShootingLayer},
        ad::SciMLBase.ADTypes.AbstractADType, vectorizer;
        loss::Union{Symbol, Expr},
        constraints = [],
        kwargs...
    )
    dynprob = CorleoneDynamicOptProblem(layer, loss, constraints...; kwargs...)
    return OptimizationProblem(dynprob, ad, vectorizer; kwargs...)
end
