module CorleoneGame

using Corleone, LuxCore, Random, ComponentArrays, OrdinaryDiffEqTsit5
using Optimization, OptimizationMOI, Ipopt, BlockSQP2, UnoSolver
using Printf
export periodic_constraints, validate_periodic_clock, Game, StateConstraint, PointCost, Options, Workspace, solve_game, initial_controls, simulate, make_layer,
    setup_layer, best_response, audit, equilibrium, run_game, run_logged, replot_outputs, constraint_norms, create_synthetic_data, player_count, player_label, subscript

"A terminal state bound, or a bound sampled along the trajectory if path=true."
Base.@kwdef struct StateConstraint
    player::Int
    index::Int
    label::String
    lower::Float64 = -Inf
    upper::Float64 = Inf
    path::Bool = false
end

"""
Two terminal inequalities for each biological state, assigned to every player.
`x0` contains physical initial values only, excluding zero-initialized accounts.
Tolerance is absolute in each physical state's units, including log-encoded stocks.
Initial values remain fixed; closure alone does not imply stability across cycles.
"""
function periodic_constraints(x0, nplayers; tolerance, log_indices = Int[])
    isfinite(tolerance) && tolerance >= 0 || error("Invalid periodicity_tolerance")
    all(isfinite, x0) || error("Periodic initial values must be finite")
    all(j -> 1 <= j <= length(x0), log_indices) || error("Invalid log-state index")
    cs = StateConstraint[]
    for i in 1:nplayers, j in eachindex(x0)
        lo, hi = x0[j] - tolerance, x0[j] + tolerance
        if j in log_indices
            lo > 0 || error("Log-state return band must be positive")
            lo, hi = log(lo), log(hi)
        end
        push!(cs, StateConstraint(; player = i, index = j, label = "!periodic lower x$(subscript(j))", lower = lo))
        push!(cs, StateConstraint(; player = i, index = j, label = "!periodic upper x$(subscript(j))", upper = hi))
    end
    return cs
end

"An externally forced cycle must end at the same forcing phase."
function validate_periodic_clock(T, period, amplitude)
    all(isfinite, (T, period, amplitude)) && T > 0 && period > 0 || error("Invalid cycle time")
    0 <= amplitude < 1 || error("Forcing amplitude must be in [0,1)")
    cycles = T / period
    isfinite(cycles) && round(cycles) >= 1 && isapprox(cycles, round(cycles); atol = 1.0e-12, rtol = 1.0e-12) ||
        error("T must be an integer multiple of forcing_period")
    return nothing
end

"A fixed-time cost on physical states, evaluated at a control-grid node."
Base.@kwdef struct PointCost{F}
    player::Int
    time::Float64
    cost::F
end

"""
N-player open-loop game, with arbitrary state and control dimensions.
Controls are matrices (component × interval); owners[j] identifies the player who controls component j.
objective(xT,player) returns a cost to MINIMIZE. For block multiple shooting,
it must be affine in accumulated quadratures to preserve stage Hessian blocks.
Quadratures start at zero and cannot enter the dynamics. decode(x) maps numerical
states to displayed states, keeping quadratures intact.
"""
Base.@kwdef struct Game{F, O, D, V}
    name::String
    state_labels::Vector{String}
    control_labels::Vector{String}
    owners::Vector{Int}
    dynamics!::F
    objective::O
    point_costs::Vector{PointCost} = PointCost[]
    constraints::Vector{StateConstraint} = StateConstraint[]
    x0::Vector{Float64}
    T::Float64
    quadratures::Vector{Int}
    lower::Vector{Float64}
    upper::Vector{Float64}
    initial::Vector{Float64}
    state_lower::Vector{Float64} = fill(-Inf, length(x0))
    state_upper::Vector{Float64} = fill(Inf, length(x0))
    gain_scales::Vector{Float64} = ones(isempty(owners) ? 0 : maximum(owners))
    running_objectives::Vector{Int} = Int[]
    decode::D = identity
    validate::V = x -> nothing
    parameters::Vector{Float64} = Float64[]
    parameter_owners::Vector{Int} = Int[]
    parameter_labels::Vector{String} = String[]
    bounds_p::Tuple{Vector{Float64}, Vector{Float64}} = (Float64[], Float64[])
    time_parameter::Int = 0 # Physical time is parameters[index] * normalized time.
    metadata::Any = nothing
end

"Evaluate terminal, accumulated, and fixed-time costs on saved states."
function trajectory_objective(g, t, states, player, parameters = g.parameters)
    value = isempty(g.parameters) ? g.objective(last(states), player) : g.objective(last(states), player, parameters)
    for c in g.point_costs
        c.player == player || continue
        k = findfirst(s -> isapprox(s, c.time; atol = 1.0e-12 * g.T, rtol = 1.0e-12), t)
        k === nothing && error("Point-cost time $(c.time) missing from trajectory")
        value += c.cost(states[k])
    end
    return value
end

player_count(g::Game) = isempty(g.owners) ? 0 : maximum(g.owners)

"Unicode numeric subscript for display labels; indices 1, 2, 3 denote players A, B, C."
subscript(i::Integer) = join(c == '-' ? '₋' : Char(Int('₀') + Int(c) - Int('0')) for c in string(i))
player_label(n::Int, player::Int) = string(Char(Int('A') + player - 1))   # n <= 3 ? string(('A','B','C')[player]) : "$player"

Base.@kwdef struct Options
    shooting::Symbol = :single
    intervals::Int = 10
    refined_intervals::Int = 40
    shooting_intervals::Int = 10
    max_rounds::Int = 200
    damping::Float64 = 0.5
    tolerance::Float64 = 2.0e-5
    feasibility_tolerance::Float64 = 1.0e-6
    max_iters::Int = 300
    ode_tolerance::Float64 = 1.0e-9
    constraint_samples::Int = 4
    audit_samples::Int = 8
    use_synthetic::Bool = false
    nlp_solver::Symbol = :auto  # Ipopt for single, BlockSQP2 for multiple shooting
    fallback_solver::Union{Nothing, Symbol} = :ipopt
    solver_options::NamedTuple = (;)
    fallback_options::NamedTuple = (;)
    extra_starts::Vector{Float64} = [0.0, 0.5, 1.0]
    audit_every_round::Bool = false # include extra starts in each round's audit
    warm_start::Bool = false
end

"Per-player primal OCP solutions, reusable across games with compatible layouts. Not thread-safe."
struct Workspace
    responses::Dict{Int, NamedTuple}
end
Workspace() = Workspace(Dict{Int, NamedTuple}())
Base.empty!(workspace::Workspace) = (empty!(workspace.responses); workspace)

layout_signature(g, o, n) = (
    o.shooting, o.shooting_intervals, n, length(g.x0),
    Tuple(g.owners), Tuple(g.quadratures), g.T, Tuple(g.parameter_owners), g.time_parameter,
)

function check_grid(o, n)
    n > 0 || error("Control grid must have positive interval count")
    return if o.shooting == :multiple
        o.shooting_intervals > 0 && n % o.shooting_intervals == 0 ||
            error("Control grid ($n intervals) must refine the multiple-shooting grid ($(o.shooting_intervals) intervals)")
    end
end

function validate(g::Game, o::Options)
    o.shooting in (:single, :multiple) || error("Choose :single or :multiple shooting")
    o.intervals > 0 && o.refined_intervals >= o.intervals && o.shooting_intervals > 0 || error("Invalid grids")
    check_grid(o, o.intervals); check_grid(o, o.refined_intervals)
    o.nlp_solver in (:auto, :ipopt, :blocksqp, :uno) || error("Unknown NLP solver")
    o.fallback_solver in (nothing, :ipopt, :uno, :blocksqp) || error("Unknown fallback solver")
    o.shooting == :single && (o.nlp_solver == :blocksqp || o.fallback_solver == :blocksqp) &&
        error("BlockSQP2 requires multiple shooting")
    all(x -> isfinite(x)&&0 <= x <= 1, o.extra_starts) || error("Extra starts must be fractions in [0,1]")
    0 < o.damping <= 1 && min(o.max_rounds, o.max_iters, o.constraint_samples, o.audit_samples) > 0 || error("Invalid algorithm settings")
    all(x -> isfinite(x)&&x > 0, (o.tolerance, o.feasibility_tolerance, o.ode_tolerance)) || error("Invalid tolerances")
    g.T > 0 && isfinite(g.T) && all(isfinite, g.x0) || error("Invalid initial state or horizon")
    nc = length(g.owners)
    nc > 0 || error("A game needs at least one control")
    all(length(v) == nc for v in (g.lower, g.upper, g.initial, g.control_labels)) || error("Control dimensions disagree")
    nplayers = player_count(g)
    sort(unique(g.owners)) == collect(1:nplayers) || error("Players must be numbered consecutively and own at least one control")
    all(isfinite, g.lower) && all(isfinite, g.upper) && all(g.lower .<= g.initial .<= g.upper) || error("Invalid control bounds")
    length(g.state_lower) == length(g.state_upper) == length(g.x0) || error("State dimensions disagree")
    all(g.state_lower .<= g.x0 .<= g.state_upper) || error("Initial state outside invariant bounds")
    length(g.gain_scales) == nplayers && all(x -> isfinite(x)&&x > 0, g.gain_scales) || error("Invalid gain scales")
    all(i -> 1 <= i <= length(g.x0), g.running_objectives) || error("Invalid running objective indices")
    length(unique(g.quadratures)) == length(g.quadratures) || error("Duplicate quadrature indices")
    all(i -> 1 <= i <= length(g.x0), g.quadratures) && all(iszero, g.x0[g.quadratures]) || error("Invalid quadratures")
    for c in g.point_costs
        1 <= c.player <= nplayers && isfinite(c.time) && 0 <= c.time <= g.T || error("Invalid point cost")
    end
    for c in g.constraints
        1 <= c.player <= nplayers && 1 <= c.index <= length(g.x0) && c.lower <= c.upper || error("Invalid constraint $(c.label)")
    end
    return
end

function check_parameters(g, parameters)
    length(parameters) == length(g.parameters) == length(g.parameter_owners) == length(g.parameter_labels) == length(g.bounds_p[1]) == length(g.bounds_p[2]) || error("Parameter dimensions disagree")
    all(isfinite, parameters) && all(g.bounds_p[1] .- 1.0e-8 .<= parameters .<= g.bounds_p[2] .+ 1.0e-8) || error("Invalid parameters")
    all(i -> 1 <= i <= player_count(g), g.parameter_owners) || error("Invalid parameter owner")
    0 <= g.time_parameter <= length(parameters) || error("Invalid time parameter")
    return g.time_parameter == 0 || g.bounds_p[1][g.time_parameter] > 0 || error("Duration must be positive")
end
physical_times(g, tr) = g.time_parameter == 0 ? tr.t : tr.parameters[g.time_parameter] .* tr.t
fitted_parameters(p, shooting) = shooting == :single ? p.p : getproperty(p, first(propertynames(p))).p

initial_controls(g::Game, n::Int) = repeat(g.initial, 1, n)
function check_controls(g, controls)
    size(controls, 1) == length(g.owners) && size(controls, 2) > 0 || error("Control dimensions disagree")
    for c in g.point_costs
        node = c.time * size(controls, 2) / g.T
        isfinite(node) && 0 <= node <= size(controls, 2) && isapprox(node, round(node); atol = 1.0e-10, rtol = 0) ||
            error("Point-cost time $(c.time) must lie on the control grid")
    end
    return all(isfinite, controls) && all(g.lower .- 1.0e-8 .<= controls .<= g.upper .+ 1.0e-8) || error("Invalid control values")
end

"Independent tight integration, restarted at each constant-control interval."
function simulate(g::Game, controls; samples = 16, parameters = g.parameters)
    samples > 0 || error("Integration samples must be positive")
    check_controls(g, controls)
    check_parameters(g, parameters)
    n = size(controls, 2)
    times, states = [0.0], [copy(g.x0)]
    for k in 1:n
        lo, hi = (k - 1) * g.T / n, k * g.T / n
        prob = ODEProblem(g.dynamics!, copy(last(states)), (lo, hi), vcat(controls[:, k], parameters))
        sol = solve(prob, Tsit5(); abstol = 1.0e-10, reltol = 1.0e-10, saveat = range(lo, hi; length = samples + 1))
        Corleone.SciMLBase.successful_retcode(sol) || error("Independent integration failed: $(sol.retcode)")
        append!(times, sol.t[2:end]); append!(states, sol.u[2:end])
    end
    x = reduce(hcat, states)
    all(isfinite, x) || error("Nonfinite trajectory")
    physical = reduce(hcat, g.decode.(states))
    g.validate(physical)
    return (; t = times, x, physical, parameters = copy(parameters), objectives = [trajectory_objective(g, times, states, i, parameters) for i in 1:player_count(g)])
end

function make_layer(g::Game, controls, player; options = Options(), parameters = g.parameters)
    validate(g, options); check_controls(g, controls)
    n = size(controls, 2)
    check_grid(options, n)
    grid = collect(range(0, g.T; length = n + 1))
    cs = [
        j => ControlParameter(
            grid[1:(end - 1)]; name = Symbol("u_", j), controls = copy(controls[j, :]),
            bounds = g.owners[j] == player ? (g.lower[j], g.upper[j]) : (copy(max.(g.lower[j], controls[j, :] .- 1.0e-8)), copy(min.(g.upper[j], controls[j, :] .+ 1.0e-8)))
        ) for j in eachindex(g.owners)
    ]
    check_parameters(g, parameters)
    plower = [g.parameter_owners[j] == player ? g.bounds_p[1][j] : parameters[j] for j in eachindex(parameters)]
    pupper = [g.parameter_owners[j] == player ? g.bounds_p[2][j] : parameters[j] for j in eachindex(parameters)]
    prob = ODEProblem(
        g.dynamics!, copy(g.x0), (0.0, g.T), vcat(zeros(length(cs)), parameters);
        abstol = options.ode_tolerance, reltol = options.ode_tolerance, saveat = g.T / n / options.constraint_samples
    )
    kwargs = (; tunable_ic = Int[], controls = cs, bounds_p = (plower, pupper), bounds_ic = (g.state_lower, g.state_upper), quadrature_indices = g.quadratures)
    return options.shooting == :single ? SingleShootingLayer(prob, Tsit5(); kwargs...) :
        MultipleShootingLayer(prob, Tsit5(), grid[1:(n ÷ options.shooting_intervals):end]; kwargs...)
end

function setup_layer(g, controls, player; options = Options(), warm = nothing, parameters = g.parameters)
    layer = make_layer(g, controls, player; options, parameters)
    ps, st = LuxCore.setup(MersenneTwister(20260908), layer)
    p = ComponentArray(ps)
    if options.shooting == :multiple
        nodes = warm === nothing ? simulate(g, controls; samples = 1, parameters).x : nothing
        idx = setdiff(eachindex(g.x0), g.quadratures)
        stride = size(controls, 2) ÷ options.shooting_intervals
        for (k, key) in enumerate(keys(ps))
            k == 1 && continue
            # Symbol indexing copies a ComponentArray block; property access gives a view.
            getproperty(p, key).u0 .= warm === nothing ? nodes[idx, 1 + (k - 1) * stride] : getproperty(warm, key).u0
            getproperty(p, key).u0 .= clamp.(getproperty(p, key).u0, g.state_lower[idx], g.state_upper[idx])
        end
    end
    return (; layer, p, st)
end

function unpack_controls(g, fitted, shooting, n)
    nc = length(g.owners)
    shooting == :single && return permutedims(reshape(collect(fitted.controls), n, nc))
    stages = propertynames(fitted)
    n % length(stages) == 0 || error("Unexpected stage count")
    perstage = n ÷ length(stages)
    all(length(fitted[key].controls) == nc * perstage for key in stages) || error("Unexpected stage control layout")
    return reduce(hcat, (permutedims(reshape(collect(fitted[key].controls), perstage, nc)) for key in stages))
end

function constraint_norms(g, trajectory, controls)
    norms = zeros(player_count(g))
    for j in eachindex(g.owners)
        norms[g.owners[j]] += sum(max.(0, g.lower[j] .- controls[j, :])) + sum(max.(0, controls[j, :] .- g.upper[j]))
    end
    for c in g.constraints
        vals = c.path ? trajectory.x[c.index, :] : [trajectory.x[c.index, end]]
        norms[c.player] += maximum(max.(0, c.lower .- vals) .+ max.(0, vals .- c.upper))
    end
    return norms
end

function solve_ocp(prob, solver, layer, st, o; attributes = o.solver_options)
    if solver == :ipopt
        opts = merge(
            (;
                tol = 1.0e-8, max_iter = o.max_iters, print_level = 0, sb = "yes",
                hessian_approximation = "limited-memory",
            ), attributes
        )
        return solve(prob, Ipopt.Optimizer(); opts...)
    elseif solver == :uno
        opts = merge((; preset = "filtersqp", max_iterations = o.max_iters, logger = "SILENT"), attributes)
        return solve(prob, UnoSolver.Optimizer(; opts...))
    end
    opts = BlockSQP2.sparse_options()
    opts.enable_premature_termination = true; opts.max_extra_steps = 1
    opts.par_QPs = false; opts.automatic_scaling = false; opts.print_level = 0; opts.feas_tol = 1.0e-9
    layout = BlockSQP2.NLPlayouts.get_layout(layer, LuxCore.initialparameters(MersenneTwister(20260908), layer), st)
    settings = merge(
        (;
            options = opts, abstol = 1.0e-8, maxiters = o.max_iters,
            blockIdx = Corleone.get_block_structure(layer), vblocks = BlockSQP2.create_vblocks(layout),
        ), attributes
    )
    return solve(prob, BlockSQP2.Optimizer(); settings...)
end

function nlp_iterations(result)
    original = hasproperty(result, :original) ? result.original : nothing
    return hasproperty(original, :solve_it) ? original.solve_it : result.stats.iterations
end

function best_response(g::Game, controls, player; options = Options(), start = nothing, workspace = nothing, parameters = g.parameters)
    nplayers = player_count(g)
    player in 1:nplayers || error("Player must be in 1:$nplayers")
    validate(g, options); check_controls(g, controls); check_grid(options, size(controls, 2))
    candidate = copy(controls)
    owned = findall(==(player), g.owners)
    signature = layout_signature(g, options, size(controls, 2))
    previous = options.warm_start && workspace !== nothing ? get(workspace.responses, player, nothing) : nothing
    compatible = previous !== nothing && previous.signature == signature
    warm = compatible && start === nothing ? previous.p : nothing
    if compatible && start === nothing
        candidate[owned, :] .= clamp.(previous.controls[owned, :], g.lower[owned], g.upper[owned])
    end
    if start !== nothing
        0 <= start <= 1 || error("Start is a fraction of the control range")
        candidate[owned, :] .= g.lower[owned] .+ start .* (g.upper[owned] .- g.lower[owned])
    end
    cpu_start = ccall(:clock, Clong, ())
    layer, p, st = setup_layer(g, candidate, player; options, warm, parameters)
    axes = getaxes(p)
    trajectory = z -> first(layer(nothing, ComponentArray(z, axes), st))
    shot = trajectory(p)
    nm = options.shooting == :multiple ? length(Corleone.shooting_constraints(shot)) : 0
    selected = filter(c -> c.player == player, g.constraints)
    lc, uc = zeros(nm), zeros(nm)
    for c in selected
        count = c.path ? length(shot.u) : 1
        append!(lc, fill(c.lower, count)); append!(uc, fill(c.upper, count))
    end
    function cons!(res, z, _)
        sol = trajectory(z)
        nm == 0 || Corleone.shooting_constraints!(view(res, 1:nm), sol)
        k = nm
        for c in selected
            for x in (c.path ? sol.u : (last(sol.u),))
                res[k += 1] = x[c.index]
            end
        end
        return res
    end
    loss = (z, _) -> begin
        sol = trajectory(z)
        trajectory_objective(g, sol.t, sol.u, player, fitted_parameters(ComponentArray(z, axes), options.shooting))
    end
    fun = isempty(lc) ? OptimizationFunction(loss, AutoForwardDiff()) : OptimizationFunction(loss, AutoForwardDiff(); cons = cons!)
    lb, ub = Corleone.get_bounds(layer) .|> ComponentArray
    prob = OptimizationProblem(fun, collect(p); lb = collect(lb), ub = collect(ub), lcons = lc, ucons = uc)
    solver = options.nlp_solver == :auto ? (options.shooting == :single ? :ipopt : :blocksqp) : options.nlp_solver
    sec_precompile = (ccall(:clock, Clong, ()) - cpu_start) / 1.0e6
    cpu_start = ccall(:clock, Clong, ())
    result = solve_ocp(prob, solver, layer, st, options)
    attempts = [(; solver, status = string(result.retcode), iterations = nlp_iterations(result))]
    fallback = options.fallback_solver
    if !Corleone.SciMLBase.successful_retcode(result) && fallback !== nothing && fallback != solver
        @warn "OCP solver did not report success; retrying" solver fallback retcode = result.retcode
        # Reuse the failed solver's finite primal iterate, projected to current bounds.
        if length(result.u) == length(p) && all(isfinite, result.u)
            prob = Corleone.SciMLBase.remake(prob; u0 = clamp.(collect(result.u), collect(lb), collect(ub)))
        end
        result = solve_ocp(prob, fallback, layer, st, options; attributes = options.fallback_options)
        solver = fallback
        push!(attempts, (; solver, status = string(result.retcode), iterations = nlp_iterations(result)))
    end
    seconds = (ccall(:clock, Clong, ()) - cpu_start) / 1.0e6
    iterations = sum(a.iterations for a in attempts)
    @printf(
        "Player %s start=%s warm=%s NLP_iterations=%d CPU_seconds=%.6f precompile=%.6f status=%s\n",
        player_label(nplayers, player), string(start), string(warm !== nothing), iterations, seconds, sec_precompile, string(result.retcode)
    )
    flush(stdout)
    #    Corleone.SciMLBase.successful_retcode(result) || error("Player $player response failed: $(result.retcode)")
    Corleone.SciMLBase.successful_retcode(result) || print("Player $player response failed: $(result.retcode)")
    newcontrols = unpack_controls(g, ComponentArray(result.u, axes), options.shooting, size(controls, 2))
    frozen = findall(!=(player), g.owners)
    maximum(abs.(newcontrols[frozen, :] .- controls[frozen, :]); init = 0.0) <= 1.0e-6 || error("Opponent policy changed")
    newcontrols .= clamp.(newcontrols, g.lower, g.upper)
    newcontrols[frozen, :] .= controls[frozen, :]
    newparameters = collect(fitted_parameters(ComponentArray(result.u, axes), options.shooting))
    for j in eachindex(newparameters)
        if g.parameter_owners[j] != player
            abs(newparameters[j] - parameters[j]) <= 1.0e-7 || error("Opponent parameter changed")
            newparameters[j] = parameters[j]
        end
    end
    check_parameters(g, newparameters)
    dense = simulate(g, newcontrols; samples = options.audit_samples, parameters = newparameters)
    norms = constraint_norms(g, dense, newcontrols)
    norms[player] <= options.feasibility_tolerance || error("Player $player fails independent feasibility check: $(norms[player])")
    optimal_shot = trajectory(result.u)
    matching = nm == 0 ? 0.0 : maximum(abs, Corleone.shooting_constraints(optimal_shot))
    matching <= options.feasibility_tolerance || error("Shooting continuity failed: $matching")
    abs(trajectory_objective(g, optimal_shot.t, optimal_shot.u, player, newparameters) - dense.objectives[player]) <= 1.0e-5 * g.gain_scales[player] || error("Shooting and independent objectives disagree")
    if options.warm_start && workspace !== nothing
        workspace.responses[player] = (; signature, controls = copy(newcontrols), p = ComponentArray(copy(result.u), axes))
    end
    return (;
        controls = newcontrols, parameters = newparameters, trajectory = dense, status = string(result.retcode), matching, iterations,
        cpu_seconds = seconds, solver, attempts, warm_started = warm !== nothing,
    )

end

function audit(g, controls; options = Options(), multistart = false, workspace = Workspace(), parameters = g.parameters)
    validate(g, options)
    trajectory = simulate(g, controls; samples = options.audit_samples, parameters)
    nplayers = player_count(g)
    gains, responses = zeros(nplayers), Any[]
    for player in 1:nplayers
        best = nothing
        best_cost = Inf
        for start in (multistart ? (nothing, options.extra_starts...) : (nothing,))
            candidate = best_response(g, controls, player; options, start, workspace, parameters)
            cost = candidate.trajectory.objectives[player]
            if cost < best_cost
                best_cost, best = cost, candidate
            end
        end
        gains[player] = max(0.0, trajectory.objectives[player] - best_cost)
        push!(responses, best)
    end
    norms = constraint_norms(g, trajectory, controls)
    return (;
        gains, normalized = gains ./ g.gain_scales, responses, trajectory, norms,
        feasible = maximum(norms) <= options.feasibility_tolerance, multistart,
    )
end

function equilibrium(g::Game; options = Options(), n = options.intervals, initial = nothing, workspace = Workspace(), parameters = g.parameters, on_round = nothing)
    validate(g, options)
    controls = initial === nothing ? initial_controls(g, n) : copy(initial)
    parameters = copy(parameters)
    size(controls, 2) == n || error("Initial policy has wrong grid")
    check_controls(g, controls); check_grid(options, n)
    history, snapshots = NamedTuple[], NamedTuple[]
    nplayers = player_count(g)
    @printf(
        "Solving game with %d players. Grid n=%d shooting=%s solver=%s\n",
        nplayers, n, string(options.shooting), options.nlp_solver
    )
    flush(stdout)
    local check # Retain the final round audit when the round limit is reached.
    for round in 1:options.max_rounds
        for player in 1:nplayers
            response = best_response(g, controls, player; options, workspace, parameters)
            owned = findall(==(player), g.owners)
            controls[owned, :] .= (1 - options.damping) .* controls[owned, :] .+ options.damping .* response.controls[owned, :]
            parameters .= (1 - options.damping) .* parameters .+ options.damping .* response.parameters
        end
        check = audit(g, controls; options, workspace, multistart = options.audit_every_round, parameters)
        push!(history, (; round, objectives = check.trajectory.objectives, gains = check.normalized, norms = check.norms))
        push!(snapshots, (; controls = copy(controls), parameters = copy(parameters), trajectory = check.trajectory))
        # Optional per-round output of the damped profile, e.g., to write partial records.
        on_round === nothing || on_round(n, last(history), last(snapshots))
        labels = join(["phi_$(player_label(nplayers, i))=$(check.trajectory.objectives[i])" for i in 1:nplayers], " ")
        @printf(
            "** Round=%d n=%d %s gains=%s violations=%s\n",
            round, n, labels, string(check.normalized), string(check.norms)
        )
        flush(stdout)
        if check.feasible && maximum(check.normalized) <= options.tolerance
            #            if !options.audit_every_round && !isempty(options.extra_starts)  # Audit is only performed on refined grid
            #                check = audit(g,controls;options,workspace,multistart=true)
            #            end
            if check.feasible && maximum(check.normalized) <= options.tolerance
                return (; controls, parameters, history, snapshots, check, trajectory = check.trajectory, converged = true)
            end
            player = argmax(check.feasible ? check.normalized : check.norms)
            controls = copy(check.responses[player].controls)
            parameters = copy(check.responses[player].parameters)
            for player in 1:nplayers
                response = best_response(g, controls, player; options, workspace, parameters)
                controls, parameters = response.controls, response.parameters
            end
        end
    end
    controls = copy(last(snapshots).controls)
    parameters = copy(last(snapshots).parameters)
    #    check = audit(g,controls;options,workspace,multistart=true)    # Audit is only performed on refined grid
    @warn "Round limit reached; no approximate equilibrium certified" game = g.name
    return (; controls, parameters, history, snapshots, check, trajectory = check.trajectory, converged = false)
end


"Transfer a piecewise-constant policy by averaging over each target interval."
function transfer_controls(controls, n)
    nc = size(controls, 2)
    nc == n && return copy(controls)
    result = zeros(size(controls, 1), n)
    # Integer coordinates avoid roundoff at shared grid boundaries.
    for j in 1:n, k in max(1, fld((j - 1) * nc, n) + 1):min(nc, cld(j * nc, n))
        overlap = max(0, min(j * nc, k * n) - max((j - 1) * nc, (k - 1) * n))
        result[:, j] .+= (overlap / nc) .* controls[:, k]
    end
    return result
end

"Solve coarse and refined forward games without producing plots or files."
function solve_game(g::Game; options = Options(), initial = nothing, workspace = Workspace(), parameters = g.parameters, on_round = nothing)
    validate(g, options)
    coarse = equilibrium(g; options, initial, workspace, parameters, on_round)
    runs = [coarse]
    refined_check = nothing
    if options.refined_intervals > options.intervals
        initial = transfer_controls(coarse.controls, options.refined_intervals)
        #        refined_check = audit(g,initial;options,workspace,multistart=true) # Additional audit is time-consuming
        push!(runs, equilibrium(g; options, n = options.refined_intervals, initial, workspace, parameters = coarse.parameters, on_round))
    end
    return (; game = g, runs, refined_check, converged = all(r.converged for r in runs))
end

include("utils.jl")
end
