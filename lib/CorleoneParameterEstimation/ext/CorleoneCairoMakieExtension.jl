module CorleoneCairoMakieExtension

using CorleoneParameterEstimation
import CorleoneParameterEstimation: get_data
using CairoMakie

# --------------------------------------------------------------------------------------
# Easy plotting of fitted states and observed data.
#
# This extension adds two CairoMakie recipes to `CorleoneParameterEstimation`, exposed
# through the plain Makie `plot`/`plot!` functions (exactly like the base `Trajectory`
# recipe), e.g.
#
#   * `plot(ax, result)` / `plot!(ax, result)` overlay the fitted model trajectories
#     (lines) with the observed data (points) of every experiment, one color per observed
#     quantity. `result` is the object returned by `fitted_parameters(pe, sol)`.
#
#   * `plot(ax, data)` / `plot!(ax, data)` plot only the observed data points, where
#     `data` is an `ExperimentData`.
#
# Both support the keyword arguments
#
#   * `idxs`: observed-quantity indices (integers) or state variable names (Symbols)
#             to plot; `nothing` (default) plots every observed quantity,
#   * `experiment_indices`: experiment indices to include; `nothing` (default) plots
#             every experiment,
#   * `labels`: optional per-quantity labels (replaces the default `q$k` / state names),
#   * `markersize = 6`, `linewidth = 2.0`: styling of the points and fitted lines.
#
# Because they share an axis, several fits or datasets can be combined in one `Figure`
# with a single `axislegend(ax)`.
#
# Like the base `Trajectory` recipe, observations of a quantity are colored with
# `Cycled(k)` (one color per observed quantity `k`), so the fitted line and its data
# always share a color.
# --------------------------------------------------------------------------------------

# Number of observed quantities of one experiment: the number of rows of the shared-grid
# matrix `y`, or the number of per-quantity series.
_observed_quantities(exp::Experiment) = exp.times === nothing ? size(exp.y, 1) : length(exp.y)

# Observed values of quantity `k` of `exp` as `(times, values)`.
function _data_series(exp::Experiment, k::Int)
    if exp.times === nothing                  # shared-grid matrix form
        return exp.t, vec(exp.y[k, :])
    end
    return exp.times[k], exp.y[k]             # per-quantity form
end

# Fitted model values of quantity `k` of experiment `exp`, evaluated at its observation
# times via the layer's `observed` map, as `(times, values)`.
function _fitted_series(pe::ParameterEstimationLayer, traj, exp::Experiment, k::Int)
    if exp.times === nothing
        return exp.t, vec(model_values(pe, traj, exp.t)[k, :])
    end
    return exp.times[k], vec(model_values(pe, traj, exp.times[k])[k, :])
end

# Resolve the `idxs` argument (integers = observed-quantity indices, or Symbols resolved
# against the state variable names of `traj`) into a vector of integer quantity indices
# covering `nq`.
function _resolve_idxs(idxs, nq::Int, traj)
    isnothing(idxs) && return collect(1:nq)
    all(i -> i isa Integer, idxs) && return collect(Int, idxs)
    isnothing(traj) && error(
        "Symbolic `idxs` require a trajectory to resolve against (pass integer quantity indices).")
    vars = traj.sys.variables
    return [vars[name] for name in idxs]
end

# Default label of observed quantity `k`: the state variable name when the observed map is
# the identity on the state (so quantity `k` is state `k`), otherwise `"q$k"`.
function _quantity_label(traj, nq::Int, k::Int)
    if !isnothing(traj)
        vars = traj.sys.variables
        name = findfirst(v -> v == k, vars)   # the Symbol whose value is `k`
        name === nothing || return string(name)
    end
    return "q$(k)"
end

# Convenience: `plot(result)` / `plot(data)` (no axis) create a Figure with sensible
# defaults -- a title and an x-label -- and plot the fit into its only axis.
# Plotting into an *existing* axis is handled by the mutating `plot!(ax, x)` recipes above.
function Makie.plot(x::ParameterEstimationResult; kwargs...)
    return plot(Figure(), x; kwargs...)
end
function Makie.plot(x::ExperimentData; kwargs...)
    return plot(Figure(), x; kwargs...)
end

# Fill in the default title / x-label only when the plot function created the axis itself,
# i.e. when the user calls `plot(result)` instead of `plot!(ax, result)`. The mutating
# `plot!` recipes (used with an explicit axis) leave the user's axis untouched.
function Makie.plot(fig::Figure, x::ParameterEstimationResult; kwargs...)
    ax = Axis(fig[1, 1]; title = "Parameter estimation result", xlabel = "Time t")
    plot!(ax, x; kwargs...)
    return fig
end
function Makie.plot(fig::Figure, x::ExperimentData; kwargs...)
    ax = Axis(fig[1, 1]; title = "Parameter estimation result", xlabel = "Time t")
    plot!(ax, x; kwargs...)
    return fig
end

# --- Recipe: observed data only --------------------------------------------------------
Makie.plottype(::ExperimentData) = Makie.Scatter

Makie.used_attributes(::Type{<:Plot}, ::ExperimentData) =
    (:idxs, :experiment_indices, :labels, :markersize)

function Makie.convert_arguments(
        ::Type{<:Plot},
        data::ExperimentData;
        idxs = nothing,
        experiment_indices = nothing,
        labels = nothing,
        markersize = 6,
    )
    exps = experiments(data)
    n_exp = length(exps)
    exp_ind = isnothing(experiment_indices) ? (1:n_exp) : experiment_indices

    specs = PlotSpec[]
    for i in exp_ind
        exp = exps[i]
        nq = _observed_quantities(exp)
        ks = _resolve_idxs(idxs, nq, nothing)
        for k in ks
            tx, yx = _data_series(exp, k)
            base = labels isa AbstractVector ? string(labels[k]) : _quantity_label(nothing, nq, k)
            pref = n_exp > 1 ? "exp $i, " : ""
            push!(specs, PlotSpec(:Scatter, Point2f.(tx, yx);
                color = Cycled(k), markersize = markersize, label = pref * base))
        end
    end
    return specs
end

# --- Recipe: fitted trajectories + observed data ---------------------------------------
Makie.plottype(::ParameterEstimationResult) = Makie.Lines

Makie.used_attributes(::Type{<:Plot}, ::ParameterEstimationResult) =
    (:idxs, :experiment_indices, :labels, :markersize, :linewidth)

function Makie.convert_arguments(
        ::Type{<:Plot},
        result::ParameterEstimationResult;
        idxs = nothing,
        experiment_indices = nothing,
        labels = nothing,
        markersize = 6,
        linewidth = 2.0,
    )
    pe = result.layer
    data = get_data(pe)
    traj = result.trajectory
    exps = experiments(data)
    n_exp = length(exps)
    exp_ind = isnothing(experiment_indices) ? (1:n_exp) : experiment_indices

    specs = PlotSpec[]
    for i in exp_ind
        exp = exps[i]
        traj_i = n_exp == 1 ? traj : traj[i]
        nq = _observed_quantities(exp)
        ks = _resolve_idxs(idxs, nq, traj_i)
        for k in ks
            base = labels isa AbstractVector ? string(labels[k]) :
                _quantity_label(traj_i, nq, k)
            pref = n_exp > 1 ? "exp $i, " : ""

            tfit, yfit = _fitted_series(pe, traj_i, exp, k)
            push!(specs, PlotSpec(:Lines, Point2f.(tfit, yfit);
                color = Cycled(k), linewidth = linewidth, label = pref * base * " (fit)"))

            tx, yx = _data_series(exp, k)
            push!(specs, PlotSpec(:Scatter, Point2f.(tx, yx);
                color = Cycled(k), markersize = markersize, label = pref * base * " (data)"))
        end
    end
    return specs
end

end
