"""
    model_values(pe::ParameterEstimationLayer, traj)
    model_values(pe::ParameterEstimationLayer, traj, times)

Evaluate the observation map of `pe` at every time point of the trajectory `traj` and
return the predicted values as a matrix of size `(n_obs, n_t)`. The data time points of
`pe` are expected to be a subset of `traj.t` (the layer is constructed with
`saveat = data.t` automatically, so this holds by default). The three-argument variant
evaluates the observation map at the given `times` instead of all of `traj.t`.

If a trajectory only partially covers the requested `times` (e.g. a shooting sub-interval
that diverged mid-solve and did not reach every observation time), the uncovered columns
are returned as `NaN` rather than aborting the calculation, so that a least-squares
optimizer sees a non-finite objective and can back off. It is only an error when *no*
requested time is present in the trajectory.
"""
function model_values(pe::ParameterEstimationLayer, traj)
    return model_values(pe, traj, traj.t)
end

function model_values(pe::ParameterEstimationLayer, traj, times::AbstractVector)
    obs = get_observed(pe)
    # Map the requested `times` onto the trajectory time points. The trajectory is built
    # with `saveat = unique(t_i)`, so the observation times are a subset of `traj.t` (the
    # solve may additionally store the end of the shared `tspan`, which is beyond the last
    # observation time of a shorter experiment and is dropped here).
    idx = indexin(times, traj.t)
    n_obs = _obs_length(obs, first(traj.u), traj.p, first(traj.t))
    Y = _alloc_obs_matrix(eltype(first(traj.u)), n_obs, length(times))
    if any(isnothing, idx)
        # A shooting sub-interval can diverge mid-solve (stiff steps or a blow-up), leaving
        # a trajectory that does not cover all observation times. Rather than aborting the
        # whole solve, mark the uncovered columns `NaN` so the least-squares objective is
        # non-finite and the optimizer backs off (line search) or rejects the step. If every
        # requested time is missing the calculation is meaningless, so that is still an error.
        all(isnothing, idx) && throw(ArgumentError(
            "The observation time points must be a subset of the trajectory's time points.")
        )
        fill!(Y, NaN)
        for (j, k) in enumerate(idx)
            isnothing(k) && continue
            _obs_into!(Y, j, obs, traj.u[k], traj.p, traj.t[k])
        end
        return Y
    end
    for (j, k) in enumerate(idx)
        _obs_into!(Y, j, obs, traj.u[k], traj.p, traj.t[k])
    end
    return Y
end

# The observation map is expected to return a tuple/vector of the measured quantities.
_obs_length(obs, u, p, t) = length(obs(u, p, t))
_alloc_obs_matrix(::Type{El}, n_obs, n_t) where {El} = Matrix{El}(undef, n_obs, n_t)
_obs_into!(Y, k, obs, u, p, t) = (Y[:, k] .= obs(u, p, t))

"""
    residuals!(res, pe::ParameterEstimationLayer, ps, st[, alg])

Compute the least-squares residuals `res = (y_obs - y_model) ./ σ` (or unweighted when the
data has no prescribed noise scale `σ`) for the parameter values `ps` and the layer state
`st`, writing them in-place into `res`. The residuals of each experiment are stacked
column-wise over the observed components first, then over the time points, and the
residuals of all experiments are concatenated in the order of the experiments.

This is the objective behind the `NonlinearLeastSquaresProblem` constructed by
[`CorleoneParameterEstimationProblem`](@ref) and can be used directly with any
NonlinearSolve least-squares solver.
"""
function residuals!(res::AbstractVector, pe::ParameterEstimationLayer, ps, st, args...)
    out, _ = pe(nothing, ps, st, args...)
    data = get_data(pe)
    n_exp = n_experiments(data)
    trajs = n_exp == 1 ? (out,) : out
    offset = 0
    for i in 1:n_exp
        exp_i = experiments(data)[i]
        chunk = view(res, (offset + 1):(offset + _n_observed(exp_i)))
        if isnothing(exp_i.times)
            # Shared-grid matrix form: every observed quantity is measured at `exp_i.t`.
            Y = model_values(pe, trajs[i], exp_i.t)
            if isnothing(exp_i.σ)
                _residuals!(chunk, exp_i.y, Y)
            else
                _residuals!(chunk, exp_i.y, Y, exp_i.σ)
            end
        else
            # Per-quantity form: quantity `k` is measured at `exp_i.times[k]`.
            if isnothing(exp_i.σ)
                _residuals_per_quantity!(chunk, pe, trajs[i], exp_i)
            else
                _residuals_per_quantity!(chunk, pe, trajs[i], exp_i, exp_i.σ)
            end
        end
        offset += _n_observed(exp_i)
    end
    # Multiple shooting: append the continuity (matching) residuals of every experiment's
    # shooting layer, so the local interval states are coupled and the reconstructed
    # trajectory is (approximately) continuous. `Corleone.shooting_constraints!` writes
    # nothing for a single-shooting trajectory, so this adds nothing when no shooting is
    # used. The total number of continuity residuals is `n_shooting_constraints(pe)`.
    for i in 1:n_exp
        n_shoot = _layer_shooting_constraints(get_layer(pe)[i])
        Corleone.shooting_constraints!(
            view(res, (offset + 1):(offset + n_shoot)), trajs[i]
        )
        offset += n_shoot
    end
    return res
end

# Residuals for the per-quantity form: for each quantity `k`, evaluate the observation map's
# `k`-th component at `times[k]` and compare against the `k`-th value series.
function _residuals_per_quantity!(res, pe, traj, exp)
    k0 = 0
    for k in eachindex(exp.times)
        yk = model_values(pe, traj, exp.times[k])[k, :]
        chunk = view(res, (k0 + 1):(k0 + length(exp.y[k])))
        copyto!(chunk, exp.y[k] .- yk)
        k0 += length(exp.y[k])
    end
    return res
end

function _residuals_per_quantity!(res, pe, traj, exp, σ_per_quantity)
    k0 = 0
    for k in eachindex(exp.times)
        yk = model_values(pe, traj, exp.times[k])[k, :]
        chunk = view(res, (k0 + 1):(k0 + length(exp.y[k])))
        σk = σ_per_quantity isa AbstractVector ? σ_per_quantity[k] : σ_per_quantity
        copyto!(chunk, (exp.y[k] .- yk) ./ σk)
        k0 += length(exp.y[k])
    end
    return res
end

# --------------------------------------------------------------------------------------
# Out-of-place residual path
# --------------------------------------------------------------------------------------
#
# The in-place `residuals!` above is the fast path used for forward-mode AD and the
# NonlinearSolve/Optimization interfaces. Reverse-mode AD (e.g. `Zygote.jl`, and hence
# `AutoZygote`/`AutoReverseDiff` through an ODE solve) cannot differentiate through the
# mutating buffer writes (`copyto!`, `view(...) .= ...`) or through the in-place
# `Corleone.shooting_constraints!`. The functions below rebuild the same residual vector
# using only functional array operations (broadcasting, `vcat`, `reduce(hcat, ...)`) so it
# can be differentiated by reverse-mode AD. The residual layout is identical to
# `residuals!`: all observation residuals first (experiment by experiment), then the
# multiple-shooting continuity residuals (experiment by experiment).

# Functional counterpart of `model_values`: build the `(n_obs, n_t)` observation matrix
# without mutating a preallocated buffer. The element type follows the trajectory state so
# that forward-mode `Dual` numbers are preserved (matching `model_values`).
function _model_values_oop(pe::ParameterEstimationLayer, traj, times::AbstractVector)
    obs = get_observed(pe)
    # `findfirst` (rather than `indexin`) keeps this branch free of array mutation, which
    # reverse-mode AD (Zygote) cannot differentiate through.
    idx = map(t -> findfirst(isequal(t), traj.t), times)
    nT = eltype(first(traj.u))
    n_obs = _obs_length(obs, first(traj.u), traj.p, first(traj.t))
    if any(isnothing, idx)
        # A shooting sub-interval can diverge mid-solve, leaving a trajectory that does not
        # cover every observation time; mark the uncovered columns `NaN` (see `model_values`).
        all(isnothing, idx) && throw(ArgumentError(
            "The observation time points must be a subset of the trajectory's time points."))
        cols = map(idx) do k
            isnothing(k) ? fill(nT(NaN), n_obs) : collect(obs(traj.u[k], traj.p, traj.t[k]))
        end
        return reduce(hcat, cols)
    end
    cols = map(idx) do k
        collect(obs(traj.u[k], traj.p, traj.t[k]))
    end
    return reduce(hcat, cols)
end

# Functional residual for one experiment (shared-grid and per-quantity forms), without the
# multiple-shooting continuity constraints.
function _experiment_obs_residual(pe::ParameterEstimationLayer, traj, exp::Experiment)
    if isnothing(exp.times)
        Y = _model_values_oop(pe, traj, exp.t)
        return _obs_residual_oop(exp.y, Y, exp.σ)
    end
    # Per-quantity form: quantity `k` is measured at `exp.times[k]`.
    blocks = map(eachindex(exp.times)) do k
        yk = _model_values_oop(pe, traj, exp.times[k])[k, :]
        σk = exp.σ isa AbstractVector ? exp.σ[k] : exp.σ
        _obs_residual_oop(exp.y[k], yk, σk)
    end
    return reduce(vcat, blocks)
end

# Functional weighted/unweighted residual of a shared-grid experiment.
function _obs_residual_oop(y_obs::AbstractMatrix, y_model::AbstractMatrix, σ)
    R = if isnothing(σ)
        y_obs .- y_model
    elseif σ isa AbstractVector
        (y_obs .- y_model) ./ reshape(σ, :, 1)
    else
        (y_obs .- y_model) ./ σ
    end
    return vec(R)
end

# Functional residual of one observed quantity (per-quantity form): `σ` is a scalar or a
# vector matching `y_obs`.
function _obs_residual_oop(y_obs::AbstractVector, y_model::AbstractVector, σ)
    R = isnothing(σ) ? (y_obs .- y_model) : (y_obs .- y_model) ./ σ
    return vec(R)
end

# Normalize the return value of a `ParameterEstimationLayer` call (a single `Trajectory`
# for one experiment, a tuple of trajectories for several) to a tuple, dispatching on the
# concrete return type rather than branching on the runtime experiment count (which would
# make the result type a `Union` and defeat inference in the out-of-place `residual`).
_layer_trajectories(traj) = (traj,)
_layer_trajectories(trajs::Tuple) = trajs

# The `p` block of the model parameters in the layer parameters `ps`. For a single
# experiment `ps` is the layer's parameter NamedTuple (single shooting: `(u0, p, controls)`;
# multiple shooting: the interval NamedTuples), while for several experiments it is a tuple
# with one entry per experiment; unwrap the tuple recursively.
_model_params(layer, ps::Tuple) = isempty(ps) ? () : _model_params(layer, first(ps))
_model_params(::Corleone.SingleShootingLayer, ps::NamedTuple) = ps.p
function _model_params(::Corleone.MultipleShootingLayer, ps::NamedTuple)
    return isempty(ps) ? () : first(values(ps)).p
end

# Element type of the residual vector. Taken from the model parameters so that manual AD
# through `residual` stays dual-safe, falling back to the state element type when the model
# has no parameter block.
function _residual_eltype(pe::ParameterEstimationLayer, ps)
    layers = get_layer(pe)
    mp = _model_params(layers[1], ps)
    return isempty(mp) ? eltype(Corleone.get_problem(layers[1]).u0) : eltype(mp)
end

"""
    residual(pe, ps, st[, alg])

Return the vector of least-squares residuals as a new array (the out-of-place counterpart of
[`residuals!`](@ref)). It uses only functional array operations, so -- unlike `residuals!` --
it can be differentiated with reverse-mode automatic differentiation. Single shooting works
with both `Zygote.jl` and `ReverseDiff.jl`; multiple shooting works with `ReverseDiff.jl`
(the root package's multiple-shooting trajectory assembly mutates, so it is not
`Zygote`-differentiable). The residual layout is identical to `residuals!`: all observation
residuals first, then the multiple-shooting continuity residuals.
"""
function residual(pe::ParameterEstimationLayer, ps, st, args...)
    out, _ = pe(nothing, ps, st, args...)
    data = get_data(pe)
    trajs = _layer_trajectories(out)
    obs_blocks = map((traj, exp) -> _experiment_obs_residual(pe, traj, exp), trajs, experiments(data))
    shoot_blocks = map(traj -> Corleone.shooting_constraints(traj), trajs)
    r = vcat(reduce(vcat, obs_blocks), reduce(vcat, shoot_blocks))
    return _concretize_residual(pe, ps, r)
end

# Single-shooting residuals are already inferred as a concrete `Vector`. A
# `MultipleShootingLayer`'s trajectory type is not fully inferred (its `shootings` NamedTuple
# is built by splatting a `Vector`), so the blocks above infer as `Any`; assert the concrete
# element type for the plain floating-point case so the residual stays stable. Other number
# types are left untouched: `ForwardDiff.Dual` does not need the hint, and for
# `ReverseDiff.TrackedReal` the residual's tape parameter need not match the parameters'
# (the assertion would wrongly reject it).
_concretize_residual(::ParameterEstimationLayer, ps, r) = r
function _concretize_residual(
        pe::ParameterEstimationLayer{<:Tuple{<:Corleone.MultipleShootingLayer, Vararg}}, ps, r
    )
    return _assert_residual_eltype(_residual_eltype(pe, ps), r)
end

_assert_residual_eltype(::Type{El}, r) where {El <: AbstractFloat} = r::Vector{El}
_assert_residual_eltype(::Type, r) = r

"""
    n_shooting_constraints(pe::ParameterEstimationLayer)

Total number of shooting continuity (matching) residuals over all experiments of `pe`,
computed from each experiment's (multiple-)shooting layer. This is zero when every
experiment uses a `SingleShootingLayer`.

!!! note
    This convenience method recomputes the layer states from the layer, which is not
type-stable for `MultipleShootingLayer`s. Prefer the `n_shooting_constraints` field of a
[`CorleoneParameterEstimationProblem`](@ref), which is computed once at construction.
"""
function n_shooting_constraints(pe::ParameterEstimationLayer)
    return sum(i -> _layer_shooting_constraints(get_layer(pe)[i]), 1:n_experiments(get_data(pe)))
end

# The number of continuity (matching) constraints generated by one experiment's shooting
# layer. Single shooting has no matching constraints; a multiple-shooting layer counts
# state, parameter and control continuity constraints in that order (see `Corleone`).
_layer_shooting_constraints(::Corleone.SingleShootingLayer) = 0
_layer_shooting_constraints(layer::Corleone.MultipleShootingLayer) =
    Corleone.get_number_of_shooting_constraints(layer)

# Type-stable variant that reuses already-computed layer parameters and states instead of
# recomputing them from the layer (which would call the not-type-stable `initialstates`).
_layer_shooting_constraints(::Corleone.SingleShootingLayer, ps, st) = 0
_layer_shooting_constraints(layer::Corleone.MultipleShootingLayer, ps, st) =
    Corleone.get_number_of_shooting_constraints(layer, ps, st)

"""
    n_residuals(pe::ParameterEstimationLayer)

Total number of least-squares residuals for `pe`: the observation residuals over all
 experiments plus any multiple-shooting continuity residuals.
"""
n_residuals(pe::ParameterEstimationLayer) =
    n_observations(get_data(pe)) + n_shooting_constraints(pe)

function _residuals!(res, y_obs::AbstractMatrix, y_model::AbstractMatrix)
    @assert length(res) == length(y_obs) == length(y_model)
    for i in eachindex(res)
        res[i] = y_obs[i] - y_model[i]
    end
    return res
end

function _residuals!(res, y_obs::AbstractMatrix, y_model::AbstractMatrix, σ)
    @assert length(res) == length(y_obs) == length(y_model)
    # Broadcast `σ` over the (n_obs × n_t) matrix: scalar, per-component vector, or full matrix.
    if σ isa AbstractVector
        R = (y_obs .- y_model) ./ reshape(σ, :, 1)
    else
        R = (y_obs .- y_model) ./ σ
    end
    copyto!(res, R)
    return res
end
