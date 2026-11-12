"""
$(TYPEDEF)

A lightweight Lux layer that connects consecutive optimal-control stages inside a
`Stages` container. It receives the `Trajectory` produced by the preceding
`ShootingLayer` and returns the initial state vector of the subsequent stage. The
mapping is encoded in the callable `f`, i.e.

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
    length(first(bounds)) == length(p) == length(last(bounds)) || throw(DimensionMismatch("The number of transition parameters and the bounds must match."))
    return StageTransition{F, typeof(p), typeof(bounds)}(f, p, bounds)
end

LuxCore.initialparameters(::Random.AbstractRNG, transition::StageTransition) = (p = copy(transition.p),)
LuxCore.initialstates(::Random.AbstractRNG, ::StageTransition) = NamedTuple()
LuxCore.parameterlength(transition::StageTransition) = length(transition.p)

# v2 recursive bounds API: return trees mirroring `initialparameters`.
get_lower_bound(transition::StageTransition, ps, st = nothing) = (p = first(transition.bounds),)
get_upper_bound(transition::StageTransition, ps, st = nothing) = (p = last(transition.bounds),)

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

so that the two phases are coupled through a constraint rather than by substitution.

The map `m` may be

- the identity (default, `StageLinkage(u0)`),
- an affine map `m(x, \\theta) = A x + B \\theta + c` (`StageLinkage(A, B, c)`), or
- an arbitrary callable `m(x, \\theta)` returning the next initial state
  (`StageLinkage(m; u0 = ...)`).

The linkage carries its own tunable parameters: the interface state `u0` (bounded by
`bounds_u0`) and, optionally, the jump parameters `\\theta` (bounded by `bounds_θ`). The
residual bounds default to `(zeros, zeros)`, i.e. an exact linkage.

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
    length(first(bounds_u0)) == n == length(last(bounds_u0)) || throw(DimensionMismatch("The interface-state bounds must match the length of `u0`."))
    length(first(bounds_θ)) == m == length(last(bounds_θ)) || throw(DimensionMismatch("The linkage-parameter bounds must match the length of `θ`."))
    length(first(bounds)) == n == length(last(bounds)) || throw(DimensionMismatch("The linkage-residual bounds must match the length of `u0`."))
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

LuxCore.initialparameters(::Random.AbstractRNG, linkage::StageLinkage) = (u0 = copy(linkage.u0), θ = copy(linkage.θ))
LuxCore.initialstates(::Random.AbstractRNG, ::StageLinkage) = NamedTuple()
LuxCore.parameterlength(linkage::StageLinkage) = length(linkage.u0) + length(linkage.θ)

get_lower_bound(linkage::StageLinkage, ps, st = nothing) = (u0 = first(linkage.bounds_u0), θ = first(linkage.bounds_θ))
get_upper_bound(linkage::StageLinkage, ps, st = nothing) = (u0 = last(linkage.bounds_u0), θ = last(linkage.bounds_θ))

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
$(TYPEDEF)

A sequence of optimal-control stages with optional [`StageTransition`](@ref) and
[`StageLinkage`](@ref) connectors. `Stages` composes multiple [`ShootingLayer`](@ref)s
without pulling `Lux.Chain` into the optimization problem: it mirrors the chain's
per-stage parameter/state bookkeeping, forwards inputs through the stages in application
order, and dispatches [`CorleoneDynamicOptProblem`](@ref) construction.

Build one by listing the stages in application order, e.g.

```julia
Stages(stage1, StageTransition((traj, p) -> last(traj.u)), stage2)
```

Parameters and states are initialized with `LuxCore.setup` into trees keyed
`stage_1, stage_2, ...` following the layer order. Calling the container
`stages(nothing, ps, st)` runs the whole sequence once and returns the final output with
the final state, like a chain would.

# Fields
$(FIELDS)
"""
struct Stages{T}
    "The stages and connectors in application order"
    layers::T

    # Restricting construction to a Tuple makes the splat constructor below apply to
    # single-layer calls as well (the default constructor would otherwise win).
    Stages(layers::Tuple) = new{typeof(layers)}(layers)
end

Stages(layers...) = Stages(layers)
Stages(layers::AbstractVector) = Stages(Tuple(layers))

# Stage parameter/state keys follow the layer order.
_multistage_keys(stages::Stages) = _multistage_keys(length(stages.layers))
_multistage_keys(n::Int) = ntuple(i -> Symbol(:stage_, i), n)

LuxCore.initialparameters(rng::Random.AbstractRNG, stages::Stages) =
    NamedTuple{_multistage_keys(stages)}(map(layer -> LuxCore.initialparameters(rng, layer), stages.layers))
LuxCore.initialstates(rng::Random.AbstractRNG, stages::Stages) =
    NamedTuple{_multistage_keys(stages)}(map(layer -> LuxCore.initialstates(rng, layer), stages.layers))
LuxCore.parameterlength(stages::Stages) = sum(LuxCore.parameterlength, stages.layers; init = 0)

function Base.show(io::IO, stages::Stages)
    print(io, "Stages(")
    join(io, stages.layers, ", ")
    return print(io, ")")
end

@inline function (stages::Stages)(inp, ps, st)
    return _multistage_call(stages.layers, 1, _multistage_keys(stages), ps, st, inp)
end

@inline _multistage_call(layers::Tuple{}, i, names, ps, st, inp) = (inp, st)
@inline function _multistage_call(layers::Tuple, i, names, ps, st, inp)
    outp, next_st = first(layers)(inp, getproperty(ps, names[i]), getproperty(st, names[i]))
    updated_st = merge(st, NamedTuple{(names[i],)}((next_st,)))
    return _multistage_call(Base.tail(layers), i + 1, names, ps, updated_st, outp)
end

"""
$(SIGNATURES)

Collect the `ShootingLayer`s contained in `stages` in application order.
"""
function _multistage_stages(stages::Stages)
    return [layer for layer in stages.layers if layer isa ShootingLayer]
end

"""
$(SIGNATURES)

Run the whole sequence of shooting stages and `StageTransition`s/`StageLinkage`s forward
once, threading the state through the transitions, and return an immutable tuple of the
`Trajectory` of every shooting stage in application order together with the linkage
residuals. This mirrors `stages(nothing, ps, st)` while additionally capturing the
intermediate trajectories so that per-stage path constraints can be evaluated.
"""
function _multistage_forward(stages::Stages, ps, st)
    return _multistage_forward(stages.layers, 1, _multistage_keys(stages), ps, st, nothing, (), ())
end

@inline _multistage_forward(layers::Tuple{}, i, names, ps, st, inp, trajs, resids) = (trajs, resids)
@inline function _multistage_forward(layers::Tuple, i, names, ps, st, inp, trajs, resids)
    layer = first(layers)
    sub_ps = getproperty(ps, names[i])
    sub_st = getproperty(st, names[i])
    if layer isa ShootingLayer
        traj, _ = layer(inp, sub_ps, sub_st)
        return _multistage_forward(Base.tail(layers), i + 1, names, ps, st, traj, (trajs..., traj), resids)
    elseif layer isa StageLinkage
        r = linkage_residual(layer, sub_ps, inp)
        outp, _ = layer(inp, sub_ps, sub_st)
        return _multistage_forward(Base.tail(layers), i + 1, names, ps, st, outp, trajs, (resids..., r))
    else
        # StageTransition or any other mapping: previous state -> next initial state.
        outp, _ = layer(inp, sub_ps, sub_st)
        return _multistage_forward(Base.tail(layers), i + 1, names, ps, st, outp, trajs, resids)
    end
end

function get_bounds(stages::Stages, ps, st = nothing)
    return get_lower_bound(stages, ps, st), get_upper_bound(stages, ps, st)
end

# v2 recursive bounds API: return trees mirroring `initialparameters`.
function get_lower_bound(stages::Stages, ps, st = nothing)
    return _multistage_bounds(get_lower_bound, stages, ps, st)
end
function get_upper_bound(stages::Stages, ps, st = nothing)
    return _multistage_bounds(get_upper_bound, stages, ps, st)
end
function get_integrality(stages::Stages, ps, st = nothing)
    return _multistage_bounds(get_integrality, stages, ps, st)
end

function _multistage_bounds(operation, stages::Stages, ps, st)
    names = _multistage_keys(stages)
    return NamedTuple{names}(
        ntuple(length(stages.layers)) do i
            return operation(stages.layers[i], getproperty(ps, names[i]), isnothing(st) ? nothing : getproperty(st, names[i]))
        end
    )
end

"""
    get_number_of_linkage_constraints(stages)

Return the total number of linkage constraints contributed by `stages` (one per component
of every `StageLinkage`'s interface state).
"""
get_number_of_linkage_constraints(::Any) = 0
get_number_of_linkage_constraints(linkage::StageLinkage) = length(linkage.u0)
get_number_of_linkage_constraints(stages::Stages) =
    sum(layer -> get_number_of_linkage_constraints(layer), stages.layers; init = 0)

"""
    linkage_constraint_bounds(stages)

Collect the lower and upper linkage-residual bounds of all `StageLinkage`s in `stages`, in
application order.
"""
linkage_constraint_bounds(::Any) = (Float64[], Float64[])
linkage_constraint_bounds(linkage::StageLinkage) = (collect(first(linkage.bounds)), collect(last(linkage.bounds)))
function linkage_constraint_bounds(stages::Stages)
    parts = map(layer -> linkage_constraint_bounds(layer), stages.layers)
    lb = reduce(vcat, map(first, parts); init = Float64[])
    ub = reduce(vcat, map(last, parts); init = Float64[])
    return (lb, ub)
end

# Neither a transition nor a linkage contributes shooting constraints.
get_number_of_shooting_constraints(::StageTransition, ps = nothing, st = nothing) = 0
get_number_of_shooting_constraints(::StageLinkage, ps = nothing, st = nothing) = 0

# Sum the shooting constraints of every shooting stage in a multistage container.
function get_number_of_shooting_constraints(stages::Stages, ps, st = nothing)
    names = _multistage_keys(stages)
    return sum(eachindex(stages.layers); init = 0) do i
        get_number_of_shooting_constraints(
            stages.layers[i],
            getproperty(ps, names[i]),
            isnothing(st) ? nothing : getproperty(st, names[i]),
        )
    end
end

"""
$(SIGNATURES)

Rewrite every bare signal symbol in `expr` into a timed call at `t`, leaving explicit
`s(t)` calls untouched. This makes an expression evaluate pointwise instead of returning
its whole saved series (so operators such as `^` apply to scalars).
"""
function _time_signal(expr, t, indices)
    if expr isa Symbol
        return haskey(indices, expr) ? Expr(:call, expr, t) : expr
    elseif expr isa Expr
        if expr.head === :call && first(expr.args) isa Symbol && haskey(indices, first(expr.args))
            return Expr(:call, first(expr.args), map(a -> _time_signal(a, t, indices), expr.args[2:end])...)
        end
        return Expr(expr.head, map(a -> _time_signal(a, t, indices), expr.args)...)
    end
    return expr
end

"""
$(SIGNATURES)

Compile `expr` into a vector-valued evaluator holding the expression evaluated at each
of the normalized times `ts` (in order). The trajectory must have saved every requested
time.
"""
function _compile_at_times(expr, ts, indices, times, eval_module, evaluator)
    components = [ _time_signal(expr, t, indices) for t in ts ]
    return compile_trajectory_expression(Expr(:vect, components...), indices, times, eval_module, evaluator)
end

"""
$(SIGNATURES)

Terminal value of a compiled trajectory expression on stage `k`.
"""
_multistage_terminal(expr, traj) = last(expr(traj))

"""
$(SIGNATURES)

Build a dynamic optimization problem from a `Stages` container of shooting layers joined
by `StageTransition`/`StageLinkage` layers. The objective is a sum of per-stage terms and
path constraints may target any stage.

# Keyword arguments
- `parameter_constraints`: a `Pair` of a function `ps -> AbstractVector` and a
  `NamedTuple` with fields `lb` and `ub`, constraining the whole container's parameters.
- `rng`: the random number generator used to initialize the stages.
"""
function CorleoneDynamicOptProblem(
        stages::Stages,
        loss, constraints...;
        rng::Random.AbstractRNG = Random.default_rng(),
        ps = nothing, st = nothing,
        parameter_constraints::Union{Nothing, Pair{<:Function, <:NamedTuple{(:lb, :ub)}}} = nothing,
        eval_module::Module = @__MODULE__,
        evaluator = default_evaluator(), kwargs...
    )
    isnothing(ps) == isnothing(st) || throw(ArgumentError("Supply initialized ps and st together."))
    if isnothing(ps)
        ps, st = LuxCore.setup(rng, stages)
    end
    stage_list = _multistage_stages(stages)
    n_stages = length(stage_list)
    n_stages > 0 || throw(ArgumentError("The provided container does not contain a ShootingLayer."))

    # Objective is a sum of per-stage terms. Each term is either a bare
    # `Symbol`/`Expr` (applied to the final stage) or a tuple `(expr, k)`.
    loss_terms = loss isa Union{Tuple, AbstractVector} ? collect(loss) : Any[loss]

    # A timed expression samples its stage trajectory at explicit normalized times. Extend
    # each stage's saved grid so the trajectory stores those samples, exactly as the
    # single-stage path does through `expression_evaluation_grid`. Extra boundaries only
    # subdivide the integration; they introduce no binding shooting constraints.
    requested = Dict{Int, Vector{Real}}()
    for term in loss_terms
        expr, k = term isa Tuple ? term : (term, n_stages)
        expr isa Expr && push!(get!(requested, k, Real[]), 1.0)
    end
    for constraint in constraints
        key, specs = constraint
        expr, k = key isa Tuple ? key : (key, n_stages)
        expr isa Union{Symbol, Expr} && append!(get!(requested, k, Real[]), specs.t)
    end
    if !isempty(requested)
        names = [name for (name, layer) in zip(_multistage_keys(stages), stages.layers) if layer isa ShootingLayer]
        for (k, ts) in requested
            sub_st = getproperty(st, names[k])
            grid = expression_evaluation_grid(stage_list[k], sub_st, Dict{Symbol, Vector{Real}}(:expr => ts))
            st = merge(st, NamedTuple{(names[k],)}((merge(sub_st, (; timestops = grid)),)))
        end
    end

    trajs, _ = _multistage_forward(stages, ps, st)

    getters = Any[]
    for term in loss_terms
        expr, k = term isa Tuple ? term : (term, n_stages)
        1 <= k <= n_stages || throw(ArgumentError("Objective stage index $k out of range (container has $n_stages shooting stages)."))
        stage = stage_list[k]
        traj = trajs[k]
        if expr isa Expr
            f = _compile_at_times(expr, (1.0,), expression_symbol_indices(stage), optimization_times(traj), eval_module, evaluator)
            push!(getters, (traj_ -> last(f(traj_)), k))
        elseif expr isa Union{Symbol, Integer}
            getter = getsym(stage.sys, expr) ∘ terminal_optimization_sample
            push!(getters, (traj_ -> getter(traj_), k))
        else
            push!(getters, (traj_ -> evaluate_trajectory_objective(expr, traj_, st), k))
        end
    end
    objective = let getters = tuple(getters...)
        function (trajs_)
            value = zero(eltype(last(trajs_[1].u)))
            for (getter, k) in getters
                value += getter(trajs_[k])
            end
            return value
        end
    end

    # Path constraints may target any stage. A bare `Symbol`/`Expr` defaults to the
    # final stage; a tuple `(expr, k)` pins it to the k-th shooting stage. Times `t`
    # are in the stage's normalized time.
    path_cons = Any[]
    path_lb = Float64[]
    path_ub = Float64[]
    for constraint in constraints
        key, specs = constraint
        expr, k = key isa Tuple ? key : (key, n_stages)
        1 <= k <= n_stages || throw(ArgumentError("Constraint stage index $k out of range (container has $n_stages shooting stages)."))
        stage = stage_list[k]
        traj = trajs[k]
        ts = collect(specs.t)
        if expr isa Union{Symbol, Expr}
            # Evaluate the expression pointwise at the requested normalized times.
            f = _compile_at_times(expr, ts, expression_symbol_indices(stage), optimization_times(traj), eval_module, evaluator)
            tidx = nothing
            series = true
            n = length(ts)
        else
            times = optimization_times(traj)
            f = expr
            series = f(traj) isa AbstractVector
            tidx = findall(t -> t in ts, times)
            n = series ? length(tidx) : 1
        end
        lb, ub = extrema(specs.bounds)
        push!(path_cons, (k, f, tidx, series))
        append!(path_lb, fill(float(lb), n))
        append!(path_ub, fill(float(ub), n))
    end

    n_shoot = sum(length(shooting_constraints(traj)) for traj in trajs; init = 0)
    link_lb, link_ub = linkage_constraint_bounds(stages)
    n_link = length(link_lb)

    param_fun = nothing
    param_lb = Float64[]
    param_ub = Float64[]
    if !isnothing(parameter_constraints)
        param_fun, param_bounds = parameter_constraints
        param_lb = collect(param_bounds.lb)
        param_ub = collect(param_bounds.ub)
        length(param_lb) == length(param_ub) || throw(DimensionMismatch("The lower and upper parameter-constraint bounds must have the same length."))
    end

    lcons = vcat(path_lb, zeros(n_shoot), link_lb, param_lb)
    ucons = vcat(path_ub, zeros(n_shoot), link_ub, param_ub)
    meta = (; path = tuple(path_cons...), param_fun)
    return CorleoneDynamicOptProblem(
        Val(:Prepared), stages, objective, meta, ps, st,
        Dict{Symbol, Vector{Real}}(), nothing, nothing, Float64[],
        lcons, ucons, n_shoot
    )
end

function optimization_objective(
        prob::CorleoneDynamicOptProblem{<:Stages},
        ps = prob.ps, st = prob.st
    )
    trajs, _ = _multistage_forward(prob.layer, ps, st)
    return scalar_objective(prob.objective(trajs))
end

function _path_series(f, traj, tidx, series)
    values = f(traj)
    if isnothing(tidx)
        return values
    elseif series
        return values[tidx]
    end
    return [values]
end

function optimization_constraints(
        prob::CorleoneDynamicOptProblem{<:Stages},
        ps = prob.ps, st = prob.st
    )
    trajs, resids = _multistage_forward(prob.layer, ps, st)
    parts = map(prob.constraints.path) do (k, f, tidx, series)
        _path_series(f, trajs[k], tidx, series)
    end
    shoot_parts = map(traj -> shooting_constraints(traj), trajs)
    param_part = isnothing(prob.constraints.param_fun) ? () : (collect(prob.constraints.param_fun(ps)),)
    return vcat(parts..., shoot_parts..., resids..., param_part...)
end

"""
$(SIGNATURES)

Write the multistage constraints into `res` in declaration order: per-stage path
constraints, shooting equalities, linkage residuals, then parameter constraints.
"""
function optimization_constraints!(
        res, prob::CorleoneDynamicOptProblem{<:Stages},
        ps = prob.ps, st = prob.st
    )
    length(res) == length(prob.lcons) || throw(DimensionMismatch("Expected $(length(prob.lcons)) constraint entries, got $(length(res))."))
    trajs, resids = _multistage_forward(prob.layer, ps, st)
    # Start at 1 rather than `firstindex(res)` so that Zygote's mutation `Buffer` (which
    # defines `setindex!` but not `firstindex`) can be used as the output.
    index = 1
    for (k, f, tidx, series) in prob.constraints.path
        values = f(trajs[k])
        if isnothing(tidx)
            for value in values
                res[index] = value
                index += 1
            end
        elseif series
            for i in tidx
                res[index] = values[i]
                index += 1
            end
        else
            res[index] = values
            index += 1
        end
    end
    for traj in trajs
        for value in shooting_constraints(traj)
            res[index] = value
            index += 1
        end
    end
    for residual in resids
        for value in residual
            res[index] = value
            index += 1
        end
    end
    if !isnothing(prob.constraints.param_fun)
        for value in prob.constraints.param_fun(ps)
            res[index] = value
            index += 1
        end
    end
    return res
end

"""
$(SIGNATURES)

Construct a `CorleoneDynamicOptProblem` and its `OptimizationProblem` facade for a
`Stages` container, accepting the multistage `loss` and per-stage `constraints`.
"""
function SciMLBase.OptimizationProblem(
        stages::Stages,
        ad::SciMLBase.ADTypes.AbstractADType, vectorizer;
        loss, constraints = (), parameter_constraints = nothing,
        rng::Random.AbstractRNG = Random.default_rng(), kwargs...
    )
    problem = CorleoneDynamicOptProblem(
        stages, loss, constraints...; rng, parameter_constraints
    )
    return SciMLBase.OptimizationProblem(problem, ad; vectorize = vectorizer, kwargs...)
end
