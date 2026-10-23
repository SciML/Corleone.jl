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
$(TYPEDEF)

A Lux layer that links consecutive optimal-control stages by an *interface state* that is
itself a free optimization variable. In contrast to [`StageTransition`](@ref), the phase
that follows a `StageLinkage` is **not** started at the value computed from the preceding
phase. Instead the linkage owns the next phase's initial state `u0` and imposes a residual

```math
\\Delta x_{\\min} \\le u_0 - m\\big(x_f^{\\mathrm{prev}}, \\theta\\big) \\le \\Delta x_{\\max}
```

so that the two phases are coupled through a constraint rather than by substitution. This
breaks the end-to-end sensitivity chain of generalized single shooting and is what makes
multiple-shooting phases scale to larger multiphase problems.

The map `m` may be

- the identity (default, `StageLinkage(u0)`),
- an affine map `m(x, \theta) = A x + B \theta + c` (`StageLinkage(A, B, c)`), or
- an arbitrary callable `m(x, \theta)` returning the next initial state
  (`StageLinkage(m; u0 = ...)`), which supports genuinely nonlinear boundary jumps such as
the dilution/reaction switch of the two-stage reactor.

The linkage carries its own tunable parameters: the interface state `u0` (bounded by
`bounds_u0`) and, optionally, the jump parameters `θ` (bounded by `bounds_θ`). The
residual bounds default to `(zeros, zeros)`, i.e. an exact linkage; pass e.g.
`bounds = ([0.0], [Inf])` for an allowed range.

# Fields
$(FIELDS)
"""
struct StageLinkage{F, U, BU, T, BT, B} <: LuxCore.AbstractLuxLayer
    "Map `(x_previous_final, θ) -> x_next_initial` used in the linkage residual"
    f::F
    "Initial values of the free interface state (the next phase's initial state)"
    u0::U
    "Lower and upper bounds `(lb, ub)` of the interface state"
    bounds_u0::BU
    "Initial values of the linkage/jump parameters"
    θ::T
    "Lower and upper bounds `(lb, ub)` of the linkage/jump parameters"
    bounds_θ::BT
    "Lower and upper bounds `(lb, ub)` of the linkage residual `u0 - m(x_f, θ)`"
    bounds::B
end

function StageLinkage(
        f::F;
        u0::AbstractVector,
        bounds_u0::Union{Nothing, Tuple{<:AbstractVector, <:AbstractVector}} = nothing,
        θ::AbstractVector = Float64[],
        bounds_θ::Union{Nothing, Tuple{<:AbstractVector, <:AbstractVector}} = nothing,
        bounds::Union{Nothing, Tuple{<:AbstractVector, <:AbstractVector}} = nothing,
    ) where {F}
    n = length(u0)
    isnothing(bounds_u0) && (bounds_u0 = (fill(-Inf, n), fill(Inf, n)))
    m = length(θ)
    isnothing(bounds_θ) && (bounds_θ = (fill(-Inf, m), fill(Inf, m)))
    isnothing(bounds) && (bounds = (zeros(n), zeros(n)))
    @assert length(first(bounds_u0)) == n == length(last(bounds_u0)) "The interface-state bounds must match the length of `u0`."
    @assert length(first(bounds_θ)) == m == length(last(bounds_θ)) "The linkage-parameter bounds must match the length of `θ`."
    @assert length(first(bounds)) == n == length(last(bounds)) "The linkage-residual bounds must match the length of `u0`."
    return StageLinkage{F, typeof(u0), typeof(bounds_u0), typeof(θ), typeof(bounds_θ), typeof(bounds)}(
        f, u0, bounds_u0, θ, bounds_θ, bounds
    )
end

"""
    StageLinkage(u0; kwargs...)

Create an identity linkage: the next phase starts at the free interface state `u0`, which
is coupled to the previous phase's final state by `u0 - x_f = 0`.
"""
function StageLinkage(u0::AbstractVector; kwargs...)
    return StageLinkage((x, θ) -> x; u0 = u0, kwargs...)
end

"""
    StageLinkage(A, B, c; u0 = c, kwargs...)

Create an affine linkage `m(x, θ) = A * x + B * θ + c`. This covers projections onto the
next phase's state (dimension changes across stages) as well as linear, tunable jumps.
"""
function StageLinkage(
        A::AbstractMatrix, B::AbstractMatrix, c::AbstractVector; u0::AbstractVector = collect(c), kwargs...
    )
    f = let A = A, B = B, c = c
        (x, θ) -> A * x .+ B * θ .+ c
    end
    return StageLinkage(f; u0 = u0, kwargs...)
end

function LuxCore.initialparameters(::Random.AbstractRNG, linkage::StageLinkage)
    return (u0 = copy(linkage.u0), θ = copy(linkage.θ))
end

LuxCore.initialstates(::Random.AbstractRNG, ::StageLinkage) = NamedTuple()
LuxCore.parameterlength(linkage::StageLinkage) = length(linkage.u0) + length(linkage.θ)

function get_bounds(linkage::StageLinkage; kwargs...)
    return (
        (u0 = first(linkage.bounds_u0), θ = first(linkage.bounds_θ)),
        (u0 = last(linkage.bounds_u0), θ = last(linkage.bounds_θ)),
    )
end

# The linkage outputs its free interface state; the residual is collected separately.
function (linkage::StageLinkage)(traj, ps, st)
    return ps.u0, st
end

"""
    linkage_residual(linkage, ps, prev_traj)

Residual `u0 - m(x_f, θ)` of a `StageLinkage`, where `x_f` is the final state of the
preceding phase's trajectory.
"""
function linkage_residual(linkage::StageLinkage, ps, prev_traj)
    return ps.u0 .- linkage.f(last(prev_traj.u), ps.θ)
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

# Neither a transition nor a linkage contributes any shooting constraints.
get_number_of_shooting_constraints(::StageTransition) = 0
get_number_of_shooting_constraints(::StageLinkage) = 0

"""
    get_number_of_linkage_constraints(layer)

Return the total number of linkage constraints contributed by `layer` (one per component of
every `StageLinkage`'s interface state).
"""
get_number_of_linkage_constraints(::Any) = 0
get_number_of_linkage_constraints(linkage::StageLinkage) = length(linkage.u0)
function get_number_of_linkage_constraints(
        layer::LuxCore.AbstractLuxWrapperLayer{:layers}
    )
    return sum(
        name -> get_number_of_linkage_constraints(layer.layers[name]),
        keys(layer.layers);
        init = 0,
    )
end

"""
    linkage_constraint_bounds(layer)

Collect the lower and upper linkage-residual bounds of all `StageLinkage`s in `layer`, in
application order.
"""
linkage_constraint_bounds(::Any) = (Float64[], Float64[])
function linkage_constraint_bounds(linkage::StageLinkage)
    return (collect(first(linkage.bounds)), collect(last(linkage.bounds)))
end
function linkage_constraint_bounds(layer::LuxCore.AbstractLuxWrapperLayer{:layers})
    parts = map(l -> linkage_constraint_bounds(l), layer.layers)
    lb = reduce(vcat, map(first, parts); init = Float64[])
    ub = reduce(vcat, map(last, parts); init = Float64[])
    return (lb, ub)
end

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

"""
$(SIGNATURES)

Like [`_forward_collect_stage_trajs`](@ref), but additionally collects the linkage residuals
of every `StageLinkage` in application order. Returns a tuple `(trajs, residuals)` where
`trajs` is the tuple of stage trajectories (as before) and `residuals` is a tuple with one
residual vector per linkage.
"""
function _forward_collect_stage_info(layer, ps::NamedTuple, st)
    return _forward_collect_stage_info(
        values(layer.layers), values(ps), values(st), nothing, (), ()
    )
end

function _forward_collect_stage_info(layer, ps, st)
    names = keys(layer.layers)
    return _forward_collect_stage_info(names, 1, layer, ps, st, nothing, (), ())
end

@inline function _forward_collect_stage_info(names, i, layer, ps, st, inp, trajs, resids)
    i > length(names) && return (trajs, resids)
    name = names[i]
    sub = getproperty(layer.layers, name)
    sub_ps = getproperty(ps, name)
    sub_st = getproperty(st, name)
    if sub isa SingleShootingLayer || sub isa MultipleShootingLayer
        traj, _ = sub(inp, sub_ps, sub_st)
        return _forward_collect_stage_info(
            names, i + 1, layer, ps, st, traj, (trajs..., traj), resids
        )
    elseif sub isa StageLinkage
        r = linkage_residual(sub, sub_ps, inp)
        outp, _ = sub(inp, sub_ps, sub_st)
        return _forward_collect_stage_info(
            names, i + 1, layer, ps, st, outp, trajs, (resids..., r)
        )
    else
        # StageTransition: Trajectory -> next initial state
        outp, _ = sub(inp, sub_ps, sub_st)
        return _forward_collect_stage_info(names, i + 1, layer, ps, st, outp, trajs, resids)
    end
end

@inline function _forward_collect_stage_info(
        ::Tuple{}, ::Tuple{}, ::Tuple{}, inp, trajs, resids
    )
    return (trajs, resids)
end

@inline function _forward_collect_stage_info(
        layers::Tuple, ps::Tuple, st::Tuple, inp, trajs, resids
    )
    isempty(layers) && return (trajs, resids)
    sub = first(layers)
    sub_ps = first(ps)
    sub_st = first(st)
    rest_layers = Base.tail(layers)
    rest_ps = Base.tail(ps)
    rest_st = Base.tail(st)
    if sub isa SingleShootingLayer || sub isa MultipleShootingLayer
        traj, _ = sub(inp, sub_ps, sub_st)
        return _forward_collect_stage_info(
            rest_layers, rest_ps, rest_st, traj, (trajs..., traj), resids
        )
    elseif sub isa StageLinkage
        r = linkage_residual(sub, sub_ps, inp)
        outp, _ = sub(inp, sub_ps, sub_st)
        return _forward_collect_stage_info(
            rest_layers, rest_ps, rest_st, outp, trajs, (resids..., r)
        )
    else
        # StageTransition: Trajectory -> next initial state
        outp, _ = sub(inp, sub_ps, sub_st)
        return _forward_collect_stage_info(rest_layers, rest_ps, rest_st, outp, trajs, resids)
    end
end

"""
    _append_shooting_constraints!(res, trajs, start)

Write the shooting/matching constraints of *every* stage trajectory in `trajs` into `res`
starting after `start`, and return the index of the last written entry. Stages without
matching conditions (single shooting) contribute nothing.
"""
function _append_shooting_constraints!(res, trajs, start::Integer)
    pos = start
    for traj in trajs
        pos = _append_shooting_constraints!(res, traj, pos)
    end
    return pos
end

function _append_shooting_constraints!(
        res, traj::Trajectory{S, U, P, T, SH}, pos::Integer
    ) where {S, U, P, T, SH <: NamedTuple}
    for kind in (:u0, :p, :controls), key in keys(traj.shooting)
        for v in traj.shooting[key][kind]
            res[pos += 1] = v
        end
    end
    return pos
end

_append_shooting_constraints!(res, traj::Trajectory, pos::Integer) = pos

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
        loss::Union{Symbol, Expr, Tuple, AbstractVector},
        constraints::Pair{<:Any, <:NamedTuple{(:t, :bounds)}}...;
        rng::Random.AbstractRNG = Random.default_rng(),
        parameter_constraints::Union{
            Nothing, Pair{<:Function, <:NamedTuple{(:lb, :ub)}},
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

    # The objective is a sum of per-stage terms. Each term is either a bare
    # `Symbol`/`Expr` (applied to the trajectory of the final stage, for backwards
    # compatibility) or a tuple `(expr, k)` pinning it to the `k`-th shooting stage.
    # This replaces the manual "carry the accumulated cost in a state" convention.
    loss_terms = loss isa Union{Tuple, AbstractVector} ? collect(loss) : Any[loss]
    getters = Tuple{Any, Int}[]
    for term in loss_terms
        expr, k = term isa Tuple ? term : (term, n_stages)
        @assert isa(expr, Symbol) || isa(expr, Expr) "The objective $(expr) is neither a symbol nor an expression!"
        @assert 1 <= k <= n_stages "Objective stage index $(k) out of range (chain has $(n_stages) shooting stages)."
        stage_st_raw = stages[k][2]
        stage_st_get = _is_ms_state(stage_st_raw) ? first(stage_st_raw) : stage_st_raw
        push!(
            getters,
            (SymbolicIndexingInterface.getsym(stage_st_get.symcache, expr), k),
        )
    end
    objective = let getters = tuple(getters...), layer = layer
        (ps, st) -> begin
            stage_trajs = _forward_collect_stage_trajs(layer, ps, st)
            val = zero(eltype(last(stage_trajs[1].u)))
            for (getter, k) in getters
                val += last(getter(stage_trajs[k]))
            end
            return val
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

    # Shooting constraints are contributed by the individual shooting layers, and the
    # linkage constraints by the `StageLinkage`s. Both are collected from *all* stages.
    n_shoot = get_number_of_shooting_constraints(layer)
    n_link = get_number_of_linkage_constraints(layer)
    link_lb, link_ub = linkage_constraint_bounds(layer)
    @assert length(link_lb) == n_link == length(link_ub)

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

    if n_path + n_shoot + n_link + n_param > 0
        constraints = let path_cons = tuple(path_cons...), layer = layer, n_path = n_path,
                n_shoot = n_shoot, n_link = n_link, param_fun = param_fun, n_param = n_param,
                n_path_cons = length(path_cons)
            (res, ps, st) -> begin
                # Run the whole chain forward once, threading the state through the
                # `StageTransition`/`StageLinkage` layers, and capture the trajectory of
                # every shooting stage plus the linkage residuals.
                stage_trajs, linkage_resids = _forward_collect_stage_info(layer, ps, st)
                # Path-constraint values are written into `res` starting at index 1
                # (the Zygote/Buffer-safe in-place convention used by
                # `_path_constraint!`; `pos` tracks 1 + cumulative count).
                pos = 0
                @inbounds for i in 1:n_path_cons
                    k, con = path_cons[i]
                    _path_constraint!(res, stage_trajs[k], con.getter, con.idx, pos + 1)
                    pos += length(con) # length(con) == length(con.idx)
                end
                if n_shoot > 0
                    # Matching conditions of *every* (multiple-)shooting stage, at an
                    # offset so the writes stay Buffer/Zygote-safe.
                    pos = _append_shooting_constraints!(res, stage_trajs, pos)
                end
                if n_link > 0
                    @inbounds for r in linkage_resids
                        for v in r
                            res[pos += 1] = v
                        end
                    end
                end
                if n_param > 0
                    @inbounds for (k, v) in enumerate(param_fun(ps))
                        res[n_path + n_shoot + n_link + k] = v
                    end
                end
                return res
            end
        end
        lcons = vcat(path_lb, zeros(n_shoot), link_lb, param_lb)
        ucons = vcat(path_ub, zeros(n_shoot), link_ub, param_ub)
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
        loss::Union{Symbol, Expr, Tuple, AbstractVector},
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
