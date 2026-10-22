# Plotting, CSV output, synthetic observations, and command-line helpers.
using CairoMakie, Statistics, Logging
using CairoMakie: Axis

# Display directives affect plots only, never model values or constraint audits.
function plot_label(label)
    label = strip(label)
    startswith(label, "!") && return nothing
    scale = 1.0
    if occursin(r"^SCALE(?:\s|$)", label)
        parts = split(label; limit = 3)
        length(parts) == 3 || error("Expected 'SCALE factor label', got '$label'")
        value = tryparse(Float64, parts[2])
        value !== nothing && isfinite(value) || error("Invalid plot scale in '$label'")
        scale = value
        label = strip(parts[3])
        isempty(label) && error("Missing display label after SCALE")
    end
    return (; label = String(label), scale)
end

function plot_entries(labels)
    entries = NamedTuple[]
    for (index, label) in enumerate(labels)
        display = plot_label(label)
        display === nothing && continue
        push!(entries, (; index, display...))
        length(entries) == 16 && break
    end
    return entries
end

# Shared bounds can occur once per player; display their common curve only once.
function constraint_plot_series(g, tr)
    series = NamedTuple[]
    seen = Set{Tuple{Int, Float64, Int, Bool, Float64}}()
    for c in g.constraints
        display = plot_label(c.label)
        display === nothing && continue
        for (bound, sign) in ((c.upper, -1), (c.lower, 1))
            c.lower == c.upper && sign == -1 && continue # An equality has one signed residual.
            isfinite(bound) || continue
            key = (c.index, bound, sign, c.path, display.scale)
            key in seen && continue
            push!(seen, key)
            players = unique(
                [
                    d.player for d in g.constraints if d.index == c.index &&
                        d.path == c.path && (sign == 1 ? d.lower : d.upper) == bound &&
                        plot_label(d.label) !== nothing && plot_label(d.label).scale == display.scale
                ]
            )
            owners = join(player_label.(Ref(player_count(g)), players), ", ")
            slack = sign .* (tr.x[c.index, :] .- bound)
            value = c.lower == c.upper ? -(c.path ? maximum(abs, slack) : abs(last(slack))) : (c.path ? minimum(slack) : last(slack))
            status = value < -1.0e-4 ? "violated" : value <= 1.0e-3 ? "active" : "inactive"
            label = "$(display.label) ($owners, $status)"
            push!(
                series, (;
                    slack = display.scale .* slack, label, path = c.path,
                    group = c.index <= length(g.state_labels) ? :states : :quadratures,
                )
            )
            length(series) == 16 && return series
        end
    end
    return series
end

function plot_limits(values; include_zero = false)
    flat = [Float64(v) for a in values for v in a if isfinite(v)]
    include_zero && push!(flat, 0.0)
    isempty(flat) && return (-1.0, 1.0)
    lo, hi = extrema(flat)
    padding = lo == hi ? 0.5 : 0.05 * (hi - lo)
    return (lo - padding, hi + padding)
end

function iteration_plot_limits(g, snapshots)
    states = plot_limits([e.scale .* tr.physical[e.index, :] for (tr, _) in snapshots for e in plot_entries(g.state_labels)])
    controls = plot_limits([e.scale .* u[e.index, :] for (_, u) in snapshots for e in plot_entries(g.control_labels)]; include_zero = true)
    series = [s for (tr, _) in snapshots for s in constraint_plot_series(g, tr)]
    slack = plot_limits([s.slack for s in series]; include_zero = true)
    return (; states, controls, slack)
end

# Wrap long entries so two legend columns fit beside an 800-wide figure.
function timeseries_legend_label(label, banks)
    banks == 1 && return label
    rows = String[""]
    for word in split(label)
        if !isempty(last(rows)) && length(last(rows)) + 1 + length(word) > 18
            push!(rows, word)
        else
            rows[end] *= (isempty(last(rows)) ? "" : " ") * word
        end
    end
    return join(rows, "\n")
end

function timeseries(g, tr, controls, path; title = g.name, ylimits = nothing)
    tplot = physical_times(g, tr)
    states = plot_entries(g.state_labels)
    inputs = plot_entries(g.control_labels)
    series = constraint_plot_series(g, tr)
    control_labels = map(inputs) do display
        owners = player_label(player_count(g), g.owners[display.index])
        label = "$(display.label) ($owners)"
        label
    end
    labels = ([e.label for e in states], control_labels, [s.label for s in series])
    banks = map(ls -> length(ls) > 8 ? 2 : 1, labels)
    legend_labels = map((ls, b) -> [timeseries_legend_label(l, b) for l in ls], labels, banks)
    # Give every axis the same actual height, including when legends wrap.
    height = maximum(
        map(
            (ls, b) -> max(
                260, cld(length(ls), b) *
                    (16 * maximum((count(==('\n'), l) + 1 for l in ls); init = 1) + 8) + 24
            ), legend_labels, banks
        )
    )
    panels = isempty(series) ? 2 : 3
    f = Figure(size = (800, panels * (height + 65) + 30))
    ax = Axis(f[1, 1]; title, ylabel = "States", height)
    for (j, e) in enumerate(states)
        lines!(ax, tplot, e.scale .* tr.physical[e.index, :]; label = legend_labels[1][j])
    end
    isempty(states) || Legend(f[1, 2], ax; nbanks = banks[1], tellwidth = false, tellheight = false, labelsize = 12)
    limits = ylimits === nothing ? plot_limits([e.scale .* tr.physical[e.index, :] for e in states]) : ylimits.states
    ylims!(ax, limits...)
    au = Axis(f[2, 1]; ylabel = "Controls", height)
    grid = range(first(tplot), last(tplot); length = size(controls, 2) + 1)
    for (j, e) in enumerate(inputs)
        values = e.scale .* controls[e.index, :]
        stairs!(au, grid, vcat(values, last(values)); step = :post, label = legend_labels[2][j])
    end
    isempty(inputs) || Legend(f[2, 2], au; nbanks = banks[2], tellwidth = false, tellheight = false, labelsize = 12)
    limits = ylimits === nothing ? plot_limits([e.scale .* controls[e.index, :] for e in inputs]; include_zero = true) : ylimits.controls
    ylims!(au, limits...)
    axes = [ax, au]
    if !isempty(series)
        ag = Axis(f[3, 1]; ylabel = "Constraint slack", height)
        hlines!(ag, [0.0]; color = :gray, linestyle = :dash)
        for (j, s) in enumerate(series)
            # Status uses the unscaled slack; terminal markers use display units.
            curve = lines!(ag, tplot, s.slack; label = legend_labels[3][j], linestyle = s.path ? :solid : :dot, linewidth = 2)
            s.path || scatter!(ag, [last(tplot)], [last(s.slack)]; color = curve.color, markersize = 10)
        end
        Legend(f[3, 2], ag; nbanks = banks[3], tellwidth = false, tellheight = false, labelsize = 12)
        limits = ylimits === nothing ? plot_limits([s.slack for s in series]; include_zero = true) : ylimits.slack
        ylims!(ag, limits...)
        push!(axes, ag)
    end
    last(axes).xlabel = "Time"
    colsize!(f.layout, 1, Relative(0.55))
    linkxaxes!(axes...)
    save(path, f)
    return f
end

function convergence_plot(g, runs, path)
    nplayers = player_count(g)
    f = Figure(size = (800, 650))
    count = sum(length(r.history) for r in runs)
    # Logarithmic axis; exact zeros (and values below ymin) are drawn at ymin.
    ymin = 1.0e-12
    ax = Axis(
        f[1, 1]; xlabel = "Best-response round", ylabel = "Normalized gain / constraint violation",
        xticks = 1:max(1, cld(count, 12)):count, yscale = log10
    )
    offset = 0
    colors = [:steelblue, :darkorange, :seagreen, :purple, :crimson, :goldenrod]
    for (j, run) in enumerate(runs)
        xs = offset .+ [h.round for h in run.history]
        for i in 1:nplayers
            label = player_label(nplayers, i)
            color = colors[mod1(i, length(colors))]
            lines!(ax, xs, [max(h.gains[i], ymin) for h in run.history]; color, label = j == 1 ? "Gain $label" : nothing)
            any(c -> c.player == i, g.constraints) &&
                lines!(ax, xs, [max(h.norms[i], ymin) for h in run.history]; color, linestyle = :dash, label = j == 1 ? "Violation $label" : nothing)
        end
        offset += length(run.history)
        j == length(runs) || vlines!(ax, [offset + 0.5]; color = :gray, linestyle = :dot)
    end
    Legend(f[1, 2], ax; tellheight = false, labelsize = 12)
    save(path, f)
    return f
end

function write_outputs(g, runs, refined_check, folder, options)
    mkpath(folder)
    result = last(runs)
    tr, controls = result.trajectory, result.controls
    tphysical = physical_times(g, tr)
    time_scale = last(tphysical) / g.T
    n = size(controls, 2)
    nplayers = player_count(g)
    headers_u = control_headers(g)
    running_headers = ["phi_$(player_label(nplayers, i))_running" for i in eachindex(g.running_objectives)]
    open(joinpath(folder, "trajectory.csv"), "w") do io
        headers = vcat(["time"], ["x_$i" for i in eachindex(g.state_labels)], running_headers, headers_u, ["numerical_x_$i" for i in eachindex(g.x0)])
        isempty(g.parameters) || append!(headers, vcat(["normalized_time"], ["parameter_$j" for j in eachindex(g.parameters)]))
        println(io, join(headers, ','))
        for k in eachindex(tr.t)
            j = min(n, floor(Int, tr.t[k] / g.T * n) + 1)
            println(
                io, join(
                    vcat(
                        tphysical[k], tr.physical[1:length(g.state_labels), k],
                        tr.x[g.running_objectives, k], controls[:, j], tr.x[:, k], isempty(g.parameters) ? Float64[] : vcat(tr.t[k], tr.parameters)
                    ), ','
                )
            )
        end
    end
    open(joinpath(folder, "controls.csv"), "w") do io
        println(io, join(vcat(["interval_start", "interval_end"], headers_u), ','))
        for k in 1:n
            println(io, join(vcat((k - 1) * g.T / n * time_scale, k * g.T / n * time_scale, controls[:, k]), ','))
        end
    end
    open(joinpath(folder, "iterations.csv"), "w") do io
        println(io, join(iteration_headers(nplayers), ','))
        for run in runs, h in run.history
            println(io, join(vcat(size(run.controls, 2), h.round, h.objectives, h.gains, h.norms), ','))
        end
    end
    open(joinpath(folder, "report.txt"), "w") do io
        println(io, "$(g.name): synthetic, uncalibrated forward game")
        println(io, "Players: $nplayers ($(join(player_label.(Ref(nplayers), 1:nplayers), ", ")))\nOptions: $options\nParameters: $(g.metadata)")
        println(io, "Objectives are minimized. Gains are positive unilateral cost decreases.")
        isempty(g.parameters) || println(io, "Optimized parameters: $(g.parameter_labels) = $(tr.parameters); physical horizon=$(last(tphysical))")
        for run in runs
            println(io, "n=$(size(run.controls, 2)), converged=$(run.converged), rounds=$(length(run.history))")
            println(io, "objectives=$(run.trajectory.objectives), normalized audited gains=$(run.check.normalized), constraint norms=$(run.check.norms)")
        end
        refined_check === nothing || println(io, "Coarse policy on refined grid: gains=$(refined_check.normalized), feasible=$(refined_check.feasible)")
        println(io, "Local deviation checks do not certify global Nash optimality.")
        println(io, "Path constraints are sampled, with denser independent validation.")
        options.shooting == :multiple && println(io, "$(options.shooting_intervals) shooting stages; BlockSQP2 uses stage Hessian blocks and variable-block metadata, without condensing.")
    end
    timeseries(g, tr, controls, joinpath(folder, "timeseries.png"))
    iteration_dir = joinpath(folder, "iterations"); mkpath(iteration_dir)
    # Keep comparable vertical scales across every saved outer-iteration plot.
    snapshots = [(s.trajectory, s.controls) for run in runs for s in run.snapshots]
    iteration_limits = iteration_plot_limits(g, snapshots)
    for file in readdir(iteration_dir)
        occursin(r"^timeseries_n\d+_iter\d+\.png$", file) && rm(joinpath(iteration_dir, file))
    end
    for run in runs, (k, s) in enumerate(run.snapshots)
        timeseries(
            g, s.trajectory, s.controls, joinpath(iteration_dir, @sprintf("timeseries_n%d_iter%03d.png", size(s.controls, 2), k));
            title = "$(g.name), n=$(size(s.controls, 2)), iteration $k", ylimits = iteration_limits
        )
    end
    convergence_plot(g, runs, joinpath(folder, "convergence.png"))
    observations = joinpath(folder, options.use_synthetic ? "synthetic.csv" : "$(lowercase(g.name))_synthetic.csv")
    return if isfile(observations)
        rows = [parse.(Float64, split(line, ',')) for line in readlines(observations)[2:end]]
        data = reduce(hcat, rows)
        size(data, 1) >= length(g.state_labels) + 1 || error("Observation CSV has too few state columns")
        f = Figure(size = (1100, 650)); ax = Axis(f[1, 1]; xlabel = "Time", ylabel = "States", title = "$(g.name): forward solution and observations")
        palette = CairoMakie.Makie.wong_colors()
        for (i, label) in enumerate(g.state_labels)
            color = palette[mod1(i, length(palette))]
            lines!(ax, tphysical, tr.physical[i, :]; color, label)
            scatter!(ax, data[1, :], data[i + 1, :]; color, label = "$label observations")
        end
        axislegend(ax; position = :rt, nbanks = 2)
        save(joinpath(folder, "forward.png"), f)
    end
end

# Read the numeric, unquoted CSV format emitted by write_outputs.
function read_output_csv(path)
    isfile(path) || error("Missing saved output: $path")
    lines = filter(s -> !isempty(strip(s)), readlines(path))
    length(lines) > 1 || error("No saved data rows in $path")
    headers = String.(split(strip(first(lines)), ','))
    length(unique(headers)) == length(headers) || error("Duplicate column names in $path")
    rows = map(enumerate(lines[2:end])) do (i, line)
        fields = split(line, ',')
        length(fields) == length(headers) || error("Wrong column count in $path, row $(i + 1)")
        values = tryparse.(Float64, fields)
        any(isnothing, values) && error("Nonnumeric value in $path, row $(i + 1)")
        Float64.(values)
    end
    return headers, reduce(hcat, rows)
end

function read_plot_outputs(g, folder)
    ih, idata = read_output_csv(joinpath(folder, "iterations.csv"))
    th, tdata = read_output_csv(joinpath(folder, "trajectory.csv"))
    column(headers, name) = something(findfirst(==(name), headers), 0)
    required(headers, name) = begin
        i = column(headers, name)
        i > 0 || error("Missing CSV column '$name'; output does not match the game")
        i
    end
    # Prefixes and file order keep CSV loading independent of display labels.
    np = player_count(g)
    gains = findall(h -> startswith(h, "normalized_gain_"), ih)
    norms = findall(h -> startswith(h, "norm_g_"), ih)
    length(gains) == length(norms) == np || error("Saved player count does not match the game")
    intervals = idata[required(ih, "intervals"), :]
    rounds = idata[required(ih, "round"), :]
    all(v -> isfinite(v) && isinteger(v) && v > 0, vcat(intervals, rounds)) || error("Invalid saved grid or round numbers")
    runs = NamedTuple[]
    for k in eachindex(rounds)
        if k == 1 || intervals[k] != intervals[k - 1] || rounds[k] <= rounds[k - 1]
            push!(runs, (; history = NamedTuple[]))
        end
        push!(last(runs).history, (; round = Int(rounds[k]), gains = idata[gains, k], norms = idata[norms, k]))
    end
    parameters = copy(g.parameters)
    for j in eachindex(parameters)
        vals = tdata[required(th, "parameter_$j"), :]
        all(isapprox.(vals, first(vals); atol = 1.0e-12, rtol = 1.0e-12)) || error("Saved parameter varies over time")
        parameters[j] = first(vals)
    end
    check_parameters(g, parameters)
    t = vec(tdata[required(th, isempty(parameters) ? "time" : "normalized_time"), :])
    if g.time_parameter != 0
        isapprox(vec(tdata[required(th, "time"), :]), parameters[g.time_parameter] .* t; atol = 1.0e-9, rtol = 1.0e-9) || error("Saved physical and normalized times disagree")
    end
    all(isfinite, t) && length(t) > 1 && all(diff(t) .> 0) || error("Saved times must be finite and strictly increasing")
    isapprox(first(t), 0; atol = 1.0e-10) && isapprox(last(t), g.T; atol = 1.0e-8 * max(1, g.T)) ||
        error("Saved time horizon does not match the game; use the original parameters")
    q = length(g.state_labels)
    physical = tdata[[required(th, "x_$i") for i in 1:q], :]
    count(h -> occursin(r"^x_\d+$", h), th) == q || error("Saved state count does not match the game")
    uc = findall(h -> startswith(h, "u_"), th)
    length(uc) == length(g.control_labels) || error("Saved control count does not match the game")
    n = Int(last(intervals))
    controls = zeros(length(uc), n)
    for j in 1:n
        # Each saved control is right-continuous. Sample inside the interval,
        # avoiding floating-point ambiguity at its boundaries.
        k = searchsortedlast(t, (j - 0.5) * g.T / n)
        k > 0 && t[k] >= (j - 1) * g.T / n - 1.0e-10 || error("Trajectory samples cannot resolve control interval $j")
        controls[:, j] = tdata[uc, k]
    end
    all(isfinite, physical) && all(isfinite, controls) || error("Nonfinite saved states or controls")
    numerical = [column(th, "numerical_x_$i") for i in eachindex(g.x0)]
    if all(>(0), numerical)
        x = tdata[numerical, :]
        all(isfinite, x) || error("Nonfinite saved numerical states")
        decoded = reduce(hcat, g.decode.(eachcol(x)))[1:q, :]
        isapprox(decoded, physical; rtol = 1.0e-8, atol = 1.0e-10) || error("Saved numerical states disagree with the current state decoder")
    elseif any(>(0), numerical)
        error("Incomplete numerical state columns in trajectory.csv")
    else
        # Older CSVs omit budget quadratures and store decoded states only.
        # Replay the saved policy once; never call the game optimizer.
        @info "Legacy trajectory CSV: replaying saved controls to recover internal states; use the original model parameters" folder
        samples = (length(t) - 1) ÷ n
        samples > 0 && samples * n + 1 == length(t) || error("Legacy trajectory does not have the expected uniform sampling")
        replay = simulate(g, controls; samples, parameters)
        isapprox(replay.t, t; rtol = 1.0e-8, atol = 1.0e-10) &&
            isapprox(replay.physical[1:q, :], physical; rtol = 1.0e-5, atol = 1.0e-7) ||
            error("Forward replay disagrees with saved states; construct the game with the original parameters")
        x = replay.x
    end
    return (; trajectory = (; t, x, physical, parameters), controls, runs)
end

"""
    replot_outputs(g; folder)

Read trajectory.csv and iterations.csv and replace only timeseries.png and
convergence.png, using current display labels. No game optimization is performed.
New CSVs include numerical states; older CSVs require one forward simulation of
saved controls to recover missing quadratures. Use the original model parameters.
"""
function replot_outputs(g::Game; folder)
    saved = read_plot_outputs(g, folder)
    timeseries(g, saved.trajectory, saved.controls, joinpath(folder, "timeseries.png"))
    convergence_plot(g, saved.runs, joinpath(folder, "convergence.png"))
    println("Updated timeseries.png and convergence.png in $folder")
    return (; timeseries = joinpath(folder, "timeseries.png"), convergence = joinpath(folder, "convergence.png"))
end

iteration_headers(nplayers) = vcat(
    ["intervals", "round"],
    ["phi_$(player_label(nplayers, i))" for i in 1:nplayers],
    ["normalized_gain_$(player_label(nplayers, i))" for i in 1:nplayers],
    ["norm_g_$(player_label(nplayers, i))" for i in 1:nplayers]
)

control_headers(g) = [
    Base.count(==(g.owners[j]), g.owners) == 1 ?
        "u_$(player_label(player_count(g), g.owners[j]))" : "u_$(player_label(player_count(g), g.owners[j]))_component_$j"
        for j in eachindex(g.owners)
]

"Write one outer round: append to iterations.csv and save the damped profile of that round."
function write_round(g, folder, n, h, snapshot)
    open(joinpath(folder, "iterations.csv"), "a") do io
        println(io, join(vcat(n, h.round, h.objectives, h.gains, h.norms), ','))
    end
    scale = g.time_parameter == 0 ? 1.0 : snapshot.parameters[g.time_parameter]
    return open(joinpath(folder, "iterations", @sprintf("controls_n%d_iter%03d.csv", n, h.round)), "w") do io
        println(io, join(vcat(["interval_start", "interval_end"], control_headers(g), ["parameter_$j" for j in eachindex(g.parameters)]), ','))
        for k in 1:n
            println(io, join(vcat((k - 1) * g.T / n * scale, k * g.T / n * scale, snapshot.controls[:, k], snapshot.parameters), ','))
        end
    end
end

function run_game(g::Game; options = Options(), folder, initial = nothing, workspace = Workspace(), parameters = g.parameters)
    # Per-round records are written during the run, so an interrupted run leaves a partial record.
    iteration_dir = joinpath(folder, "iterations"); mkpath(iteration_dir)
    for file in readdir(iteration_dir)
        occursin(r"^controls_n\d+_iter\d+\.csv$", file) && rm(joinpath(iteration_dir, file))
    end
    open(io -> println(io, join(iteration_headers(player_count(g)), ',')), joinpath(folder, "iterations.csv"), "w")
    on_round = (n, h, snapshot) -> write_round(g, folder, n, h, snapshot)
    result = solve_game(g; options, initial, workspace, parameters, on_round)
    write_outputs(g, result.runs, result.refined_check, folder, options)
    return result
end

function _read_numeric_trajectory(path)
    lines = readlines(path)
    rows = [parse.(Float64, split(line, ',')) for line in lines[2:end] if !isempty(strip(line))]
    isempty(rows) && error("No trajectory rows in $path")
    return [r[1] for r in rows], reduce(hcat, [r[2:end] for r in rows])
end

function _linear_interpolate(t, x, tq)
    tq <= first(t) && return x[:, 1]
    tq >= last(t) && return x[:, end]
    k = searchsortedlast(t, tq); α = (tq - t[k]) / (t[k + 1] - t[k])
    return (1 - α) .* x[:, k] .+ α .* x[:, k + 1]
end

function create_synthetic_data(
        g::Game, folder; sigma = 0.02, seed = 20260909,
        measurement_times = nothing, output = "synthetic.csv"
    )
    candidates = [joinpath(folder, "trajectory.csv"), joinpath(dirname(folder), "output", "trajectory.csv"), joinpath(dirname(folder), "output_multiple", "trajectory.csv")]
    existing = filter(isfile, candidates); isempty(existing) && error("No trajectory.csv found in output or output_multiple"); source = argmax([stat(f).mtime for f in existing]); source_path = existing[source]
    t, values = _read_numeric_trajectory(source_path); nstate = length(g.state_labels)
    measurement_times === nothing && (measurement_times = collect(range(first(t), last(t); length = 20)))
    size(values, 1) >= nstate || error("trajectory.csv has too few state columns")
    σ = sigma isa Number ? fill(Float64(sigma), nstate) : collect(Float64, sigma)
    length(σ) == nstate || error("sigma must be scalar or one value per displayed state")
    rng = MersenneTwister(seed); out = joinpath(folder, output); mkpath(folder)
    open(out, "w") do io
        println(io, join(vcat("time", ["x_$i" for i in 1:nstate], "sigma", "seed"), ','))
        for tq in measurement_times
            truth = _linear_interpolate(t, values[1:nstate, :], tq); obs = truth .+ σ .* randn(rng, nstate)
            println(io, join(vcat(tq, obs, mean(σ), seed), ','))
        end
    end
    println("Wrote $(length(measurement_times)) measurements to $out from $source_path")
    return out
end

"Copy Julia and native-library stdout/stderr to the original stdout and a log file."
function with_run_log(f, logpath)
    tee = Sys.which("tee")
    tee === nothing && error("Dual-output logging requires the system 'tee' utility on PATH")
    # A separate process drains the pipe even during a blocking native solver call.
    # An @async Julia reader could stall on a single-threaded runtime.
    flush(stdout); flush(stderr); Base.Libc.flush_cstdio()
    return open(pipeline(`$tee -- $logpath`; stdout = stdout, stderr = stderr), "w") do process
        redirect_stdio(; stdout = process.in, stderr = process.in) do
            # ConsoleLogger otherwise retains its original stderr stream.
            logger = ConsoleLogger(process.in, Logging.min_enabled_level(current_logger()))
            with_logger(logger) do
                try
                    return f()
                catch err
                    showerror(stderr, err, catch_backtrace())
                    println(stderr)
                    rethrow()
                finally
                    Base.Libc.flush_cstdio()
                    flush(stdout); flush(stderr)
                end
            end
        end
    end
end

function run_logged(main, directory; args = ARGS)
    length(args) <= 1 || error("Usage: case.jl [single|multiple|synthetic]")
    mode = isempty(args) ? :single : Symbol(only(args))
    mode in (:single, :multiple, :synthetic) || error("Choose single, multiple, or synthetic")
    folder = joinpath(directory, mode == :multiple ? "output_multiple" : "output")
    mkpath(folder)
    logpath = joinpath(folder, "run.log")
    return with_run_log(logpath) do
        println("Running $mode; log: $logpath")
        if mode == :synthetic
            result = main(; synthetic = true, folder = folder)
            println("Finished synthetic observations; outputs: $folder")
        else
            result = main(; shooting = mode, folder = folder)
            println("Finished; converged=$(result.converged); outputs: $folder")
        end
        return result
    end
end
