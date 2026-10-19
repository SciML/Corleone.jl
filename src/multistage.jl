"""
$(TYPEDEF)

A lightweight Lux layer that connects consecutive optimal-control stages inside a
`Lux.Chain`. It receives the `Trajectory` produced by the preceding
`SingleShootingLayer` and returns the initial state vector of the subsequent
stage. The mapping is encoded in the callable `f`, i.e.

```julia
StageTransition((traj, p) -> vcat(last(traj.u)[1:2], 0.0); p = [0.5], bounds = ([0.0], [1.0]))
```

The transition may carry its own parameters (for example a jump size or a switching
cost). They are optimized together with the parameters of the shooting layers and are
bounded by `bounds`. The callable `f` receives the preceding stage's `Trajectory` and
the transition parameters `p`, and returns the initial state of the next stage.

# Fields
$(FIELDS)
"""
struct StageTransition{F, P, B} <: LuxCore.AbstractLuxLayer
    "Function mapping the preceding stage's `Trajectory` and the parameters to the next stage's initial state"
    f::F
    "Initial values of the transition parameters"
    p::P
    "Lower and upper bounds `(lb, ub)` of the transition parameters"
    bounds::B
end

function StageTransition(
        f::F;
        p::AbstractVector = Float64[],
        bounds::Union{Nothing, Tuple{<:AbstractVector, <:AbstractVector}} = nothing,
    ) where {F}
    if isnothing(bounds)
        bounds = (fill(-Inf, length(p)), fill(Inf, length(p)))
    end
    @assert length(first(bounds)) == length(p) == length(last(bounds)) "The number of transition parameters and the bounds must match."
    return StageTransition{F, typeof(p), typeof(bounds)}(f, p, bounds)
end

function LuxCore.initialparameters(::Random.AbstractRNG, transition::StageTransition)
    return (p = copy(transition.p),)
end

LuxCore.initialstates(::Random.AbstractRNG, ::StageTransition) = NamedTuple()
LuxCore.parameterlength(transition::StageTransition) = length(transition.p)

function get_bounds(transition::StageTransition; kwargs...)
    return (p = first(transition.bounds),), (p = last(transition.bounds),)
end

function (transition::StageTransition)(traj, ps, st)
    return transition.f(traj, ps.p), st
end

"""
    get_bounds(layer::LuxCore.AbstractLuxWrapperLayer{:layers}; kwargs...)

Recursively collect the optimization-variable bounds of all sublayers of a `Lux.Chain`.
The returned named tuples mirror the parameter structure produced by
`LuxCore.initialparameters`, so that they can be flattened together with the parameters
by the vectorizer (e.g. `ComponentArrays`).
"""
function get_bounds(layer::LuxCore.AbstractLuxWrapperLayer{:layers}; kwargs...)
    names = keys(layer.layers)
    bounds = map(name -> get_bounds(layer.layers[name]; kwargs...), names)
    return NamedTuple{names}(first.(bounds)), NamedTuple{names}(last.(bounds))
end

"""
    get_number_of_shooting_constraints(layer::LuxCore.AbstractLuxWrapperLayer{:layers})

Return the total number of shooting/matching constraints contributed by all sublayers of
a `Lux.Chain`.
"""
function get_number_of_shooting_constraints(layer::LuxCore.AbstractLuxWrapperLayer{:layers})
    return sum(
        name -> get_number_of_shooting_constraints(layer.layers[name]),
        keys(layer.layers);
        init = 0,
    )
end

# A transition does not contribute any shooting constraints.
get_number_of_shooting_constraints(::StageTransition) = 0

"""
$(SIGNATURES)

Collect all `SingleShootingLayer`s contained in `layer` in the order in which they are
applied, together with their corresponding states. This is used to locate the final stage
of a multi-stage problem, which determines the objective and the time grid.
"""
_collect_shooting_stages(::Any, st) = Tuple[]
_collect_shooting_stages(layer::SingleShootingLayer, st) = [(layer, st)]

function _collect_shooting_stages(
        layer::LuxCore.AbstractLuxWrapperLayer{:layers}, st
    )
    stages = Tuple[]
    for name in keys(layer.layers)
        append!(
            stages,
            _collect_shooting_stages(layer.layers[name], getproperty(st, name)),
        )
    end
    return stages
end

"""
$(SIGNATURES)

Construct a dynamic optimization problem from a `Lux.Chain` of shooting layers. The
objective is taken from the **last** shooting stage in the chain, since the
`StageTransition` layers are expected to carry the accumulated cost of the preceding
stages into the final one. Path constraints passed positionally are imposed on the
trajectory of the final stage.

# Keyword arguments
- `parameter_constraints`: a `Pair` of a function `ps -> AbstractVector` and a
  `NamedTuple` with fields `lb` and `ub`. The function receives the concatenated
  parameters of the whole chain and its result is constrained element-wise between `lb`
  and `ub`. This makes it possible to couple quantities of different stages, for example
  to require that the stage durations sum to a given total.
- `rng`: the random number generator used to initialize the layers.
"""
function CorleoneDynamicOptProblem(
        layer::LuxCore.AbstractLuxWrapperLayer{:layers},
        loss::Union{Symbol, Expr},
        constraints::Pair{<:Any, <:NamedTuple{(:t, :bounds)}}...;
        rng::Random.AbstractRNG = Random.default_rng(),
        parameter_constraints::Union{
            Nothing, Pair{<:Function, <:NamedTuple{(:lb, :ub)}}
        } = nothing,
        kwargs...
    )
    st = LuxCore.initialstates(rng, layer)
    stages = _collect_shooting_stages(layer, st)
    @assert !isempty(stages) "The provided layer does not contain a SingleShootingLayer."
    _, final_st = last(stages)
    sys = final_st.symcache
    objective = let getter = SymbolicIndexingInterface.getsym(sys, loss), layer = layer
        (ps, st) -> begin
            traj, _ = layer(nothing, ps, st)
            last(getter(traj))
        end
    end

    # Path constraints are imposed on the trajectory of the final stage.
    tpoints = reduce(vcat, _collect_tspans(final_st.tspans...))
    path_cons = TrajectoryConstraint[]
    path_lb = Float64[]
    path_ub = Float64[]
    foreach(constraints) do (expr, specs)
        @assert isa(expr, Symbol) || isa(expr, Expr) "The constraint $(expr) is neither a symbol nor an expression!"
        tidx = findall(∈(specs.t), tpoints)
        if !isempty(tidx)
            push!(
                path_cons,
                TrajectoryConstraint(
                    SymbolicIndexingInterface.getsym(sys, expr), tidx
                ),
            )
            lb_, ub_ = extrema(specs.bounds)
            append!(path_lb, fill(lb_, length(tidx)))
            append!(path_ub, fill(ub_, length(tidx)))
        end
    end
    conseval = TrajectoryConstraintEvaluator(Tuple(path_cons))
    n_path = length(path_lb)

    # Shooting constraints are contributed by the individual shooting layers.
    n_shoot = get_number_of_shooting_constraints(layer)

    # Parameter constraints act directly on the (concatenated) parameter vector, which
    # makes it possible to couple quantities of different stages, e.g. their durations.
    param_fun = nothing
    param_lb = Float64[]
    param_ub = Float64[]
    if !isnothing(parameter_constraints)
        param_fun, param_bounds = parameter_constraints
        param_lb = collect(param_bounds.lb)
        param_ub = collect(param_bounds.ub)
        @assert length(param_lb) == length(param_ub) "The lower and upper parameter-constraint bounds must have the same length."
    end
    n_param = length(param_lb)

    if n_path + n_shoot + n_param > 0
        constraints = let conseval = conseval, layer = layer, n_path = n_path,
                          n_shoot = n_shoot, param_fun = param_fun, n_param = n_param
            (res, ps, st) -> begin
                traj, _ = layer(nothing, ps, st)
                conseval(res, traj, 0) # Path constraints (written directly, no views)
                if n_shoot > 0
                    # Shooting constraints watch the full `res` at an offset so the
                    # writes stay Buffer/Zygote-safe (no `@views` slicing).
                    shooting_constraints!(res, traj, n_path)
                end
                if n_param > 0
                    @inbounds for (k, v) in enumerate(param_fun(ps))
                        res[n_path + n_shoot + k] = v
                    end
                end
                return res
            end
        end
        lcons = vcat(path_lb, zeros(n_shoot), param_lb)
        ucons = vcat(path_ub, zeros(n_shoot), param_ub)
    else
        constraints = lcons = ucons = nothing
    end
    return CorleoneDynamicOptProblem{typeof(layer), Nothing, typeof(objective), typeof(constraints), typeof(lcons)}(
        layer, nothing, objective, constraints, lcons, ucons
    )
end

function SciMLBase.OptimizationProblem(
        layer::LuxCore.AbstractLuxWrapperLayer{:layers},
        ad::SciMLBase.ADTypes.AbstractADType, vectorizer;
        loss::Union{Symbol, Expr},
        constraints = [],
        parameter_constraints = nothing,
        kwargs...
    )
    dynprob = CorleoneDynamicOptProblem(
        layer, loss, constraints...;
        parameter_constraints = parameter_constraints, kwargs...
    )
    return OptimizationProblem(dynprob, ad, vectorizer; kwargs...)
end
