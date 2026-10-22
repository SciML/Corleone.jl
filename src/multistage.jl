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
    # `map` over the `NamedTuple` itself keeps both the keys and the element types
    # inferable. The previous `map(name -> get_bounds(layer.layers[name]), keys(...))`
    # indexed by a runtime `Symbol`, which made the per-layer result a `Union` and the
    # whole function infer `Any` for a multistage `Chain`.
    bounds = map(l -> get_bounds(l; kwargs...), layer.layers)
    return map(first, bounds), map(last, bounds)
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
# A MultipleShootingLayer wraps a SingleShootingLayer and shares its dynamics: expose the
# inner layer (with the per-interval states of the wrapped stage) so multistage chains
# built from MultipleShootingLayers can impose per-stage path constraints and get an
# objective time grid.
_collect_shooting_stages(layer::MultipleShootingLayer, st) = [(layer.layer, st)]

# A MultipleShootingLayer's state is a NamedTuple with one entry per shooting interval
# (`interval_i` keys); a plain stage state carries the interval fields directly.
_is_ms_state(st::NamedTuple) =
    !isempty(st) && all(k -> startswith(string(k), "interval_"), keys(st))
_is_ms_state(::Any) = false

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

Run the whole `Lux.Chain` of shooting layers and `StageTransition`s forward once,
threading the state through the transitions, and return an immutable tuple of the
`Trajectory` of every shooting stage in application order (matching the ordering of
`_collect_shooting_stages`). The trajectory of the final stage is the last element.

This mirrors the forward pass that `layer(nothing, ps, st)` performs, but additionally
captures the intermediate stage trajectories so that per-stage path constraints can
be evaluated. The recursion builds the result as an immutable tuple (no in-place array
mutation) and writes into the caller-provided `res` buffer via `_path_constraint!`,
so it stays Zygote/Buffer safe exactly like the single-stage path-constraint path.
"""
function _forward_collect_stage_trajs(layer, ps::NamedTuple, st)
    # Recurse structurally over the tuple of sublayers (with their matching parameter and
    # state tuples) instead of indexing `layer.layers` by a runtime `Symbol`. Runtime
    # `getproperty(::NamedTuple, ::Symbol)`/`getindex(::Tuple, ::Int)` yield a `Union` of
    # the element types, which made this function infer `Any` for a multistage `Chain`.
    return _forward_collect_stage_trajs(
        values(layer.layers), values(ps), values(st), nothing, ()
    )
end

# Fallback for container parameters that are not `NamedTuple`s (notably the
# `ComponentVector`s produced by the ComponentArrays extension, whose `values` returns the
# array itself). Indexing by a runtime `Symbol` is not type stable, but it keeps that path
# working; the `NamedTuple` (plain-Lux) path above is the type-stable one.
function _forward_collect_stage_trajs(layer, ps, st)
    names = keys(layer.layers)
    return _forward_collect_stage_trajs(names, 1, layer, ps, st, nothing, ())
end

@inline function _forward_collect_stage_trajs(names, i, layer, ps, st, inp, trajs)
    i > length(names) && return trajs
    name = names[i]
    sub = getproperty(layer.layers, name)
    sub_ps = getproperty(ps, name)
    sub_st = getproperty(st, name)
    if sub isa SingleShootingLayer || sub isa MultipleShootingLayer
        traj, _ = sub(inp, sub_ps, sub_st)
        return _forward_collect_stage_trajs(
            names, i + 1, layer, ps, st, traj, (trajs..., traj)
        )
    else
        # StageTransition: Trajectory -> next initial state
        outp, _ = sub(inp, sub_ps, sub_st)
        return _forward_collect_stage_trajs(names, i + 1, layer, ps, st, outp, trajs)
    end
end

@inline function _forward_collect_stage_trajs(
        ::Tuple{}, ::Tuple{}, ::Tuple{}, inp, trajs
    )
    return trajs
end

@inline function _forward_collect_stage_trajs(
        layers::Tuple, ps::Tuple, st::Tuple, inp, trajs
    )
    sub = first(layers)
    sub_ps = first(ps)
    sub_st = first(st)
    rest_layers = Base.tail(layers)
    rest_ps = Base.tail(ps)
    rest_st = Base.tail(st)
    if sub isa SingleShootingLayer
        traj, _ = sub(inp, sub_ps, sub_st)
        return _forward_collect_stage_trajs(
            rest_layers, rest_ps, rest_st, traj, (trajs..., traj)
        )
    elseif sub isa MultipleShootingLayer
        traj, _ = sub(inp, sub_ps, sub_st)
        return _forward_collect_stage_trajs(
            rest_layers, rest_ps, rest_st, traj, (trajs..., traj)
        )
    else
        # StageTransition: Trajectory -> next initial state
        outp, _ = sub(inp, sub_ps, sub_st)
        return _forward_collect_stage_trajs(rest_layers, rest_ps, rest_st, outp, trajs)
    end
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
    n_stages = length(stages)
    _, final_st_raw = last(stages)
    # A MultipleShootingLayer's state is a NamedTuple of per-interval states; each interval
    # state carries the same `symcache` (they are built from one and the same inner layer),
    # so unwrap to the first interval's state. For a bare SingleShootingLayer the state
    # already is the interval state.
    final_st = _is_ms_state(final_st_raw) ? first(final_st_raw) : final_st_raw
    sys = final_st.symcache
    objective = let getter = SymbolicIndexingInterface.getsym(sys, loss), layer = layer
        (ps, st) -> begin
            traj, _ = layer(nothing, ps, st)
            last(getter(traj))
        end
    end

    # Path constraints may target *any* stage. Each constraint key is either a bare
    # `Symbol`/`Expr` (state, imposed on the trajectory of the final stage, for
    # backwards compatibility) or a tuple `(expr, k)` pinning it to the `k`-th
    # shooting stage of the chain. The time points `t` are in the *normalized*
    # time `[0, 1]` of that stage.
    path_cons = Tuple{Int, TrajectoryConstraint}[]
    path_lb = Float64[]
    path_ub = Float64[]
    foreach(constraints) do (key, specs)
        expr, k = key isa Tuple ? key : (key, n_stages)
        @assert isa(expr, Symbol) || isa(expr, Expr) "The constraint $(expr) is neither a symbol nor an expression!"
        @assert 1 <= k <= n_stages "Stage index $(k) out of range (chain has $(n_stages) shooting stages)."
        stage_st_raw = stages[k][2]
        is_ms = _is_ms_state(stage_st_raw)
        stage_st_get = is_ms ? first(stage_st_raw) : stage_st_raw
        # For an MS stage the concatenated trajectory's saved times are the union of the
        # segment endpoints and the control grid points; the tspan endpoints come first
        # in `_sequential_solve`'s output, so rebuild that layout: [t_lo, grid points of
        # interval 1 without its final point, t of interval 2 start = interval 1 end,
        # grid points of interval 2 without final point, ..., t_final].
        tk = if is_ms
            layout = Float64[]
            n_int = length(stage_st_raw)
            for i in 1:n_int
                # `tspans` is a Tuple of (lo, hi) pairs for this interval; each pair is
                # one shooting segment solved by `_solve_segment`.
                for seg in getfield(stage_st_raw, i).tspans
                    append!(layout, _saved_interval_times(seg...))
                end
            end
            sort!(unique!(layout))
        else
            collect(_collect_tspans(stage_st_raw.tspans...))
        end
        tidx = findall(∈(specs.t), tk)
        if !isempty(tidx)
            push!(
                path_cons,
                (
                    k,
                    TrajectoryConstraint(
                        SymbolicIndexingInterface.getsym(stage_st_get.symcache, expr), tidx
                    ),
                ),
            )
            lb_, ub_ = extrema(specs.bounds)
            append!(path_lb, fill(lb_, length(tidx)))
            append!(path_ub, fill(ub_, length(tidx)))
        end
    end
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
        constraints = let path_cons = tuple(path_cons...), layer = layer, n_path = n_path,
                          n_shoot = n_shoot, param_fun = param_fun, n_param = n_param,
                          n_path_cons = length(path_cons)
            (res, ps, st) -> begin
                # Run the whole chain forward once, threading the state through the
                # `StageTransition` layers, and capture the trajectory of every
                # shooting stage so per-stage path constraints can be evaluated.
                stage_trajs = _forward_collect_stage_trajs(layer, ps, st)
                # Path-constraint values are written into `res` starting at index 1
                # (the Zygote/Buffer-safe in-place convention used by
                # `_path_constraint!`; `pos` tracks 1 + cumulative count).
                pos = 1
                @inbounds for i in 1:n_path_cons
                    k, con = path_cons[i]
                    _path_constraint!(res, stage_trajs[k], con.getter, con.idx, pos)
                    pos += length(con) # length(con) == length(con.idx)
                end
                traj = stage_trajs[n_stages]
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
