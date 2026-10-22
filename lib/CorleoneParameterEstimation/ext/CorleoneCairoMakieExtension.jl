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

# Fitted model values of quantity `k`, evaluated over the trajectory's *full* time grid
# `traj.t`. The layer is built with `saveat = unique(t_i)`, so `traj.t` is a superset of
# every observation time and spans the whole experiment horizon -- unlike the per-quantity
# observation times (`exp.times[k]`), which can be sparse. Using `traj.t` keeps the fitted
# *line* a genuine, continuous curve (starting at `t0`) instead of only drawing through the
# observed points, which is what happens when per-quantity data is observed sparsely.
function _fitted_series(pe::ParameterEstimationLayer, traj, k::Int)
    return traj.t, vec(model_values(pe, traj)[k, :])
end


# Resolve the `idxs` argument (integers = observed-quantity indices, or Symbols resolved
# against the state variable names of `traj`) into a vector of integer quantity indices
# covering `nq`.
function _resolve_idxs(idxs, nq::Int, traj)
    isnothing(idxs) && return collect(1:nq)
    all(i -> i isa Integer, idxs) && return collect(Int, idxs)
    isnothing(traj) && error(
        "Symbolic `idxs` require a trajectory to resolve against (pass integer quantity indices)."
    )
    vars = traj.sys.variables
    return [vars[name] for name in idxs]
end

# Default label of observed quantity `k`: the state variable name when the observed map is
# the identity on the state (so quantity `k` is state `k`), otherwise `"q$k"`.
function _quantity_label(traj, nq::Int, k::Int)
    if !isnothing(traj)
        for (name, idx) in traj.sys.variables
            idx == k && return string(name)
        end
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
            push!(
                specs, PlotSpec(
                    :Scatter, Point2f.(tx, yx);
                    color = Cycled(k), markersize = markersize, label = pref * base
                )
            )
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

            tfit, yfit = _fitted_series(pe, traj_i, k)
            push!(
                specs, PlotSpec(
                    :Lines, Point2f.(tfit, yfit);
                    color = Cycled(k), linewidth = linewidth, label = pref * base * " (fit)"
                )
            )

            tx, yx = _data_series(exp, k)
            push!(
                specs, PlotSpec(
                    :Scatter, Point2f.(tx, yx);
                    color = Cycled(k), markersize = markersize, label = pref * base * " (data)"
                )
            )
        end
    end
    return specs
end

# --------------------------------------------------------------------------------------
# Profile-likelihood confidence intervals
# --------------------------------------------------------------------------------------
#
# `plot(est)` / `plot!(ax, est)` for a `ProfileLikelihoodEstimate`: one axis per profiled
# parameter (rows of a `Figure` when several are selected), each showing the minimized
# residual-sum-of-squares profile as a function of the parameter value, together with
#
#   * the likelihood-ratio threshold (dashed black line) whose crossing defines the CI,
#   * the point estimate and its profile CI endpoints (band + vertical lines),
#   * the asymptotic CI endpoints for cross-checking (dotted lines) when they are finite.
#
# Keyword arguments:
#   * `idxs = nothing`: parameter indices (integers) or names (Symbols) to plot; `nothing`
#     plots every parameter,
#   * `fig_title = "Profile likelihood"`, `axis_label = ...`: axis texts,
#   * `threshold_color/linestyle`, `markersize = 4`, `linewidth = 2.0`: styling.
# --------------------------------------------------------------------------------------

Makie.plottype(::ProfileLikelihoodEstimate) = Makie.Lines

# One subplot (axis) for one parameter's profile.
function _profile_axis(
        fig::Figure, idx::Tuple{Int, Int}, est::ProfileLikelihoodEstimate,
        k::Integer; idxcol = 1, linewidth = 2.0, markersize = 4.0, alpha = 0.15,
        threshold_color = :black, threshold_linestyle = :dash,
        asymptotic_linestyle = :dot, axis_label = ("Parameter value", "Residual sum of squares"),
    )
    ax = Axis(fig[idx...])
    curve = est.profiles[k]
    name = isnothing(curve.name) ? string("x[", k, "]") : curve.name
    ax.title = string("Profile of ", name)
    isnothing(axis_label) || (ax.xlabel = axis_label[1]; ax.ylabel = axis_label[2])

    # A concrete color (from Makie's default Wong palette) rather than `Cycled`, because
    # filled `band!`s need a resolved RGB(A) and cannot consume a `Cycled` attribute.
    col = Makie.wong_colors()[idxcol]

    lines!(
        ax, Point2f.(curve.values, curve.rss);
        color = col, linewidth = linewidth, label = "profile"
    )
    hlines!(
        ax, [curve.threshold];
        color = threshold_color, linestyle = threshold_linestyle, label = "threshold"
    )

    # finite CI endpoints -> shading band + vertical lines
    lo, hi = curve.lower, curve.upper
    if isfinite(lo) && isfinite(hi) && lo != hi
        band!(
            ax, [lo, hi], [curve.min_rss, curve.min_rss],
            [curve.threshold, curve.threshold]; color = (col, alpha)
        )
        vlines!(ax, [lo, hi]; color = (col, 0.5), linestyle = :solid)
    end
    scatter!(
        ax, [curve.values[argmin(curve.rss)]], [curve.min_rss];
        color = col, markersize = markersize, label = "optimum"
    )

    # asymptotic CI endpoints for cross-checking (dotted), when finite and nondegenerate
    aci = est.asymptotic_intervals[k]
    if isfinite(aci[1]) && isfinite(aci[2]) && aci[1] != aci[2]
        vlines!(
            ax, [aci[1], aci[2]]; color = :gray, linestyle = asymptotic_linestyle,
            label = "asymptotic CI"
        )
    end
    return ax
end

# `plot(est)` without a user axis: build a Figure with sensible defaults and one axis per
# selected parameter, laid out in a single column.
function Makie.plot(x::ProfileLikelihoodEstimate; kwargs...)
    return plot(Figure(), x; kwargs...)
end

function Makie.plot(
        fig::Figure, est::ProfileLikelihoodEstimate; idxs = nothing,
        fig_title = "Profile likelihood", kwargs...
    )
    p = length(est.profiles)
    ks = _resolve_profile_idxs(idxs, est, p)
    ncol = min(3, length(ks))
    nrow = cld(length(ks), ncol)
    # One axis per profiled parameter, laid out in rows 1:(nrow) so that the top row
    # (row 0) is free for the figure title and the bottom row for the shared legend.
    axes = [(i, j) for i in 1:nrow, j in 1:ncol]
    first_ax = nothing
    for (q, k) in enumerate(ks)
        ax = _profile_axis(fig, axes[q], est, k; kwargs...)
        q == 1 && (first_ax = ax)
    end
    # Title spans the full top row once every axis is in place (so `fig[0, :]` is
    # meaningful after layout settles).
    isnothing(fig_title) || Label(fig[0, :], fig_title; fontsize = 18)
    # A single, shared legend below all axes (all profiled parameters use the same set of
    # labeled elements, so the first axis' legend entries describe the whole figure). This
    # keeps the legend out of the subplots, unlike an `axislegend` inside one axis.
    Legend(
        fig[nrow + 1, :], first_ax; orientation = :horizontal, framevisible = false,
        title = "Shared legend"
    )
    resize_to_layout!(fig)
    return fig
end

# Resolve `idxs` (integers or parameter-name Symbols) into integer parameter indices.
function _resolve_profile_idxs(idxs, est::ProfileLikelihoodEstimate, p::Int)
    isnothing(idxs) && return collect(1:p)
    all(i -> i isa Integer, idxs) && return collect(Int, idxs)
    names = est.names
    isnothing(names) && error(
        "Symbolic `idxs` require parameter `names` (pass integer parameter indices)."
    )
    return [findfirst(==(Symbol(n)), names) for n in idxs]
end

end
