"""
    model_values(pe::ParameterEstimationLayer, traj; sym = :default)

Evaluate the observation map of `pe` at every time point of the trajectory `traj` and
return the predicted values as a matrix of size `(n_obs, n_t)`. The data time points of
`pe` are expected to be a subset of `traj.t` (the layer is constructed with
`saveat = data.t` automatically, so this holds by default).

# Keywords
- `sym`: The `SymbolicIndexingInterface` symbol cache to resolve `traj`. When `nothing`
  the observation map is called directly on the raw state values of the trajectory.
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
    any(isnothing, idx) && throw(ArgumentError(
        "The observation time points must be a subset of the trajectory's time points.")
    )
    n_obs = _obs_length(obs, first(traj.u), traj.p, first(traj.t))
    Y = _alloc_obs_matrix(eltype(first(traj.u)), n_obs, length(times))
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

# Return a copy of the residuals (out-of-place variant).
"""
    residual(pe, ps, st[, alg])

Return the vector of least-squares residuals (a copy of [`residuals!`](@ref)).
"""
function residual(pe::ParameterEstimationLayer, ps, st, args...)
    eT = isempty(ps.p) ? eltype(first(get_layer(pe)).problem.u0) : eltype(first(ps.p))
    buffer = Vector{eT}(undef, n_observations(get_data(pe)))
    return residuals!(buffer, pe, ps, st, args...)
end

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
