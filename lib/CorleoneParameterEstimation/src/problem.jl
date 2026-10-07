"""
    CorleoneParameterEstimationProblem <: SciMLBase.AbstractNonlinearProblem

A lightweight container tying a [`ParameterEstimationLayer`](@ref) to a flat vector of
optimization variables (the tunable parameters and initial conditions of the wrapped
shooting layer) so that the least-squares residuals can be minimized with any
NonlinearSolve solver.

For a plain parameter-estimation setup (no tunable initial conditions, no controls) the
optimization variables are exactly the tunable model parameters. Parameters that are fixed
on the wrapped layer (e.g. provided via `bounds_p` with coincident lower/upper bounds) are
not part of the optimization variables and are kept at their values.

With a [`MultipleShootingLayer`](@ref) the flat vector additionally contains the *local*
shooting initial conditions of every experiment (see [`ParameterEstimationLayer`](@ref)),
which keep the trajectory continuous at the shooting points.

# Fields
$(FIELDS)

# Constructors
- `CorleoneParameterEstimationProblem(layer; rng, x0, shooting_scale)`

The low-level `NonlinearSolve.NonlinearLeastSquaresProblem` can be built directly via
[`problem`](@ref). Most users construct that problem through the convenience constructor
`NonlinearSolve.NonlinearLeastSquaresProblem(pe::ParameterEstimationLayer; kwargs...)`.
"""
struct CorleoneParameterEstimationProblem{L, P, S, St, X, B, R, R2}
    "The wrapped parameter-estimation layer"
    layer::L
    "Structured initial parameters of the first experiment (`(u0, p, controls)` for single shooting, or an `(interval_1, ...)` NamedTuple for multiple shooting)"
    ps0::P
    "Tuple of the layer parameters of every experiment (used to expand the shared optimization variables per experiment for multiple shooting)"
    ps0s::S
    "The layer state"
    st::St
    "Flat vector of optimization variables"
    x0::X
    "Flat lower bounds of the optimization variables (or `nothing`)"
    lb::B
    "Flat upper bounds of the optimization variables (or `nothing`)"
    ub::B
    "In-place residual function `(res, x, p) -> res` reconstructing the structured layer parameters from `x`"
    residual::R
    "Out-of-place residual function `(x, p) -> r` (reverse-mode-AD compatible), see [`residual`](@ref)"
    residual_oop::R2
    "Multiplier applied to the appended shooting (continuity) residuals so they can be weighted relative to the observation residuals"
    shooting_scale::Float64
    "Total number of residuals (observations plus multiple-shooting continuity constraints)"
    n_residuals::Int
    "Number of multiple-shooting continuity residuals (the last `n_shooting_constraints` entries of the residual vector)"
    n_shooting_constraints::Int
end

# --------------------------------------------------------------------------------------
# Optimization-variable layout
# --------------------------------------------------------------------------------------
#
# The flat vector `x` of optimization variables is built from the layer parameters of each
# experiment, dispatching on the kind of shooting layer:
#
# * **Single shooting**: `ps = (u0, p, controls)` is shared between all experiments, and
#   `x = [u0; p]` -- the tunable initial conditions followed by the tunable model
#   parameters.
# * **Multiple shooting**: `ps = (interval_1 = ..., ..., interval_K = ...)`, one block per
#   shooting interval, and `x = [u0; p; λ_1; ...; λ_E]` -- the shared tunable initial
#   conditions `u0`, the shared tunable model parameters `p` (interval 1), and then, per
#   experiment `e`, the *local* shooting initial conditions `λ_e` of its intervals `2..K`.
#
# The `[u0; p]` block is the *shared model* block that [`estimate_parameters`](@ref)
# reports; the local shooting states `λ_e` are auxiliary continuity variables used only to
# keep the trajectory continuous at the shooting points.

"True if `pe` wraps `MultipleShootingLayer`s."
_is_multiple_shooting(pe::ParameterEstimationLayer) =
    first(get_layer(pe)) isa Corleone.MultipleShootingLayer

# The shared `(u0, p)` model block of the (first) layer parameters `ps0`.
function _shared_block(::Corleone.SingleShootingLayer, ps0::NamedTuple)
    return (u0 = ps0.u0, p = ps0.p)
end
function _shared_block(::Corleone.MultipleShootingLayer, ps0::NamedTuple)
    interval_1 = first(values(ps0))
    return (u0 = interval_1.u0, p = interval_1.p)
end

# The local shooting initial conditions of one experiment, i.e. the concatenation of the
# `u0` blocks of its intervals `2..K`. Empty for a single-shooting layer.
function _local_states(::Corleone.SingleShootingLayer, ps0_i::NamedTuple)
    return eltype(ps0_i.p)[]
end
function _local_states(::Corleone.MultipleShootingLayer, ps0_i::NamedTuple)
    return reduce(vcat, (vec(b.u0) for b in Iterators.drop(values(ps0_i), 1)))
end

"Flat vector of the optimization variables for the layer parameters `ps0s`."
function _flatten(pe::ParameterEstimationLayer, ps0s::Tuple)
    layers = get_layer(pe)
    shared = _shared_block(layers[1], ps0s[1])
    x = reduce(vcat, (vec(v) for v in values(shared)))   # [u0; p]
    for i in eachindex(ps0s)
        append!(x, _local_states(layers[i], ps0s[i]))
    end
    return x
end

# Number of entries of `x` belonging to the shared `[u0; p]` model block.
function _n_shared(layer::ParameterEstimationLayer, ps0s::Tuple)
    shared = _shared_block(first(get_layer(layer)), ps0s[1])
    return length(shared.u0) + length(shared.p)
end

# Reconstruct the structured layer parameters `ps` used by the layer callable (and the
# residual evaluation) from the flat vector `x`.
#
# * Single shooting: a single shared `(u0, p, controls)` NamedTuple (controls picked up
#   from the first experiment and overridden per experiment with fixed design inputs).
# * Multiple shooting with one experiment: a single `(interval_1, ..., interval_K)`
#   NamedTuple carrying that experiment's controls and local states.
# * Multiple shooting with several experiments: a *tuple* with one such interval NamedTuple
#   per experiment (each with its own local shooting states and fixed controls, sharing the
#   model `[u0; p]`).
function _unflatten(pe::ParameterEstimationLayer, ps0s::Tuple, x::AbstractVector)
    layers = get_layer(pe)
    data = get_data(pe)
    n_exp = n_experiments(data)
    shared = _shared_block(layers[1], ps0s[1])
    nu = length(shared.u0)
    m = nu + length(shared.p)
    @assert length(x) >= m
    if !_is_multiple_shooting(pe)
        return (
            u0 = collect(x[1:nu]),
            p = collect(x[(nu + 1):m]),
            controls = ps0s[1].controls,
        )
    end
    # Split the flat local-states block into per-experiment slices and rebuild the interval
    # parameters of every experiment. Local states of intervals `2..K` each hold the full
    # (non-quadrature) state vector, so their size is constant across those intervals.
    ps = ntuple(n_exp) do i
        li = _n_local_states(layers[i], ps0s[i])
        start = m + 1 + (i - 1) * li
        λi = collect(x[start:(start + li - 1)])
        _unflatten_ms(ps0s[i], (u0 = collect(x[1:nu]), p = collect(x[(nu + 1):m])), λi)
    end
    return n_exp == 1 ? only(ps) : ps
end

# Number of local shooting-state entries of one experiment (0 for single shooting).
_n_local_states(::Corleone.SingleShootingLayer, ps0_i::NamedTuple) = 0
function _n_local_states(::Corleone.MultipleShootingLayer, ps0_i::NamedTuple)
    return sum(length(b.u0) for b in Iterators.drop(values(ps0_i), 1))
end

# Rebuild the `(interval_1, ..., interval_K)` NamedTuple of one multiple-shooting layer
# from the shared model block `(u0, p)` and the flat local states `λ` of its intervals
# `2..K`. Fixed controls are carried over from the initial layer parameters.
function _unflatten_ms(ps0_i::NamedTuple, shared, λ::AbstractVector)
    nint = length(ps0_i)
    blocks = ntuple(nint) do k
        base = ps0_i[k]
        if k == 1
            (u0 = copy(shared.u0), p = copy(shared.p), controls = base.controls)
        else
            s = length(ps0_i[2].u0)
            # local-state block occupied by interval `k` (intervals `2..K` are concatenated)
            lo = (k - 2) * s + 1
            hi = (k - 1) * s
            (u0 = collect(λ[lo:hi]), p = copy(shared.p), controls = base.controls)
        end
    end
    return NamedTuple{keys(ps0_i)}(blocks)
end

# --- Bounds ------------------------------------------------------------------------
function _tunable_bounds(pe::ParameterEstimationLayer, ps0s::Tuple)
    layers = get_layer(pe)
    if _is_multiple_shooting(pe)
        return _ms_bounds(pe, layers, ps0s)
    end
    return _ss_bounds(layers[1], ps0s)
end

# Single shooting: bounds of the shared `[u0; p]` block. The `u0`/`p` bound vectors
# returned by `Corleone.get_bounds` already match the sizes of `ps0.u0`/`ps0.p` (the
# initial-condition bounds are restricted to the tunable entries by the layer).
function _ss_bounds(layer::Corleone.SingleShootingLayer, ps0s::Tuple)
    ps0 = ps0s[1]
    lb, ub = Corleone.get_bounds(layer)
    flat_lb, flat_ub = eltype(ps0.p)[], eltype(ps0.p)[]
    if !isempty(ps0.u0)
        append!(flat_lb, vec(lb.u0))
        append!(flat_ub, vec(ub.u0))
    end
    if !isempty(ps0.p)
        append!(flat_lb, vec(lb.p))
        append!(flat_ub, vec(ub.p))
    end
    return flat_lb, flat_ub
end

# Multiple shooting: bounds of the shared `[u0; p]` block (from interval 1 of the first
# experiment) followed by the local shooting-state bounds of every experiment's intervals
# `2..K`.
function _ms_bounds(pe::ParameterEstimationLayer, layers::Tuple, ps0s::Tuple)
    ps0 = ps0s[1]
    el = eltype(ps0[1].p)
    flat_lb, flat_ub = el[], el[]

    lb, ub = Corleone.get_bounds(layers[1])
    if !isempty(ps0[1].u0)
        append!(flat_lb, vec(lb[1].u0))
        append!(flat_ub, vec(ub[1].u0))
    end
    if !isempty(ps0[1].p)
        append!(flat_lb, vec(lb[1].p))
        append!(flat_ub, vec(ub[1].p))
    end

    for i in eachindex(ps0s)
        lbi, ubi = Corleone.get_bounds(layers[i])
        for k in 2:length(lbi)
            append!(flat_lb, vec(lbi[k].u0))
            append!(flat_ub, vec(ubi[k].u0))
        end
    end
    return flat_lb, flat_ub
end

# --------------------------------------------------------------------------------------
# Problem construction
# --------------------------------------------------------------------------------------
function CorleoneParameterEstimationProblem(
        layer::ParameterEstimationLayer;
        rng::Random.AbstractRNG = Random.default_rng(),
        x0 = nothing,
        shooting_scale::Real = 1.0,
    )
    # One structured parameter set per experiment (its shooting layer's initial
    # parameters), plus the tuple of per-experiment layer states. `ps0` (the first
    # experiment's parameters) is the shared model block used by the accessors.
    n_exp = n_experiments(get_data(layer))
    ps0s = ntuple(i -> LuxCore.initialparameters(rng, get_layer(layer)[i]), n_exp)
    st = ntuple(i -> LuxCore.initialstates(rng, get_layer(layer)[i]), n_exp)
    x0 = isnothing(x0) ? _flatten(layer, ps0s) : x0
    lb, ub = _tunable_bounds(layer, ps0s)

    # Optional weighting of the shooting (continuity) residuals. Observations are
    # residualized against their noise scale `σ`, so they are `O(1)` at the noise floor,
    # while the appended shooting residuals are raw state/parameter differences with the
    # state scale. For multiple shooting the reconstruction is only meaningfully continuous
    # when the optimizer drives the shooting residuals to (nearly) zero. If the optimal
    # solution still has a large shooting residual, increase `shooting_scale` and re-solve:
    # this weighs the continuity residuals more heavily so the optimizer satisfies them.
    # The first `n_obs` residuals are the (measured) observations, the remaining
    # `n_shooting_constraints` are the appended shooting residuals. Both counts are computed
    # once here (reusing the already-constructed layer states, which is type-stable) and
    # stored as concrete `Int` fields so downstream code does not have to recompute them
    # from the layer (which would call the not-type-stable `initialstates`).
    n_obs::Int = n_observations(get_data(layer))
    n_shoot::Int = sum(
        i -> _layer_shooting_constraints(get_layer(layer)[i], ps0s[i], st[i]), 1:n_exp
    )
    n_res = n_obs + n_shoot
    residual = let layer = layer, ps0s = ps0s, st = st, n_obs = n_obs,
            shooting_scale = Float64(shooting_scale)
        (res, x, p) -> begin
            ps = _unflatten(layer, ps0s, x)
            residuals!(res, layer, ps, st)
            if shooting_scale != 1.0
                @views res[(n_obs + 1):end] .*= shooting_scale
            end
            return res
        end
    end
    # Out-of-place counterpart of `residual` used by the solvers when reverse-mode AD is
    # requested (see `residual` in `residuals.jl`). It reuses the functional residual path and
    # applies the same `shooting_scale` weighting without mutating in place.
    residual_oop = let layer = layer, ps0s = ps0s, st = st, n_obs = n_obs, n_shoot = n_shoot,
            shooting_scale = Float64(shooting_scale)
        (x, p) -> begin
            ps = _unflatten(layer, ps0s, x)
            r = CorleoneParameterEstimation.residual(layer, ps, st)
            if shooting_scale == 1.0
                return r
            end
            return vcat(r[1:n_obs], r[(n_obs + 1):(n_obs + n_shoot)] .* shooting_scale)
        end
    end

    return CorleoneParameterEstimationProblem(
        layer, ps0s[1], ps0s, st, x0, lb, ub, residual, residual_oop,
        Float64(shooting_scale), n_res, n_shoot
    )
end

"""
    NonlinearSolve.NonlinearLeastSquaresProblem(problem::CorleoneParameterEstimationProblem)

Build the `NonlinearLeastSquaresProblem` (with optional `lb`/`ub`) driving the least-squares
fit of the layer's tunable parameters.
"""
function SciMLBase.NonlinearLeastSquaresProblem(problem::CorleoneParameterEstimationProblem)
    # Use the stored out-of-place residual so NonlinearSolve can accurately infer the
    # residual dimension (number of observations + shooting constraints), which generally
    # differs from the number of fitted parameters. Being fully functional, it also allows
    # reverse-mode AD backends (e.g. `AutoZygote`) to differentiate the residual.
    (; residual_oop, x0, lb, ub) = problem
    return NonlinearLeastSquaresProblem(
        NonlinearFunction{false}(residual_oop), x0; lb = lb, ub = ub
    )
end

"""
    NonlinearSolve.NonlinearLeastSquaresProblem(pe::ParameterEstimationLayer; kwargs...)

Construct the `NonlinearLeastSquaresProblem` for a [`ParameterEstimationLayer`](@ref) so it
can be solved with a NonlinearSolve least-squares solver, e.g.
`solve(prob, GaussNewton())` or `solve(prob, LevenbergMarquardt())`. Additional keywords are
forwarded to [`CorleoneParameterEstimationProblem`](@ref).
"""
function SciMLBase.NonlinearLeastSquaresProblem(
        pe::ParameterEstimationLayer; kwargs...
    )
    return NonlinearLeastSquaresProblem(CorleoneParameterEstimationProblem(pe; kwargs...))
end

"""
    problem(pe::ParameterEstimationLayer; kwargs...)

Return the `NonlinearSolve.NonlinearLeastSquaresProblem` for `pe` (an alias for
`NonlinearSolve.NonlinearLeastSquaresProblem(pe; kwargs...)`).
"""
problem(pe::ParameterEstimationLayer; kwargs...) =
    NonlinearLeastSquaresProblem(pe; kwargs...)
