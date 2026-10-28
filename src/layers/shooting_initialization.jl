# Initialization is an explicit operation on existing parameter buffers, separate
# from Lux setup and the differentiable forward path.
function shooting_initialization_entries(layer::ShootingLayer, ps, st, variables)
    selected = collect(variables isa Integer ? (variables,) : variables)
    all(i -> i in Base.OneTo(length(layer.initialcondition.initialconditions)), selected) ||
        throw(ArgumentError("Shooting variable indices must refer to problem states."))
    return [
        (;
            index = i,
            values = getproperty(ps.initialcondition.initialconditions, key).parameter.ps,
            grid = getproperty(st.initialcondition.initialconditions, key).lookup.collection,
        )
            for (i, key) in enumerate(keys(layer.initialcondition.initialconditions))
            if layer.initialcondition.initialconditions[key] isa InitialCondition && i in selected
    ]
end

initialization_value(values::AbstractDict, i) = values[i]
initialization_value(values::Tuple{Vararg{Pair}}, i) = Dict(values)[i]
initialization_value(values::AbstractVector{<:Pair}, i) = Dict(values)[i]
initialization_value(values, i) = values[i]

function shooting_initialization_bounds(layer, i)
    symbol = only(filter(sym -> variable_index(layer.sys, sym) == i, variable_symbols(layer.sys)))
    return get(default_bounds(layer.sys), symbol, (-Inf, Inf))
end

function set_shooting_guesses!(layer, entries, guess)
    # Validate every proposal before changing any parameter buffer.
    proposals = map(entries) do entry
        grid = sort!(collect(entry.grid); by = last)
        proposed = [guess(entry.index, t, j) for (t, j) in grid]
        bound = shooting_initialization_bounds(layer, entry.index)
        validate_annotation_default(proposed, bound...)
        converted = similar(entry.values)
        for ((_, j), value) in zip(grid, proposed)
            converted[j] = value
        end
        converted
    end
    foreach((entry, proposed) -> copyto!(entry.values, proposed), entries, proposals)
    return nothing
end

"""
$(SIGNATURES)

Omit `ps` and `st` to run `LuxCore.setup(rng, layer)` and return initialized `(ps, st)`.

Fill tunable shooting values in `ps` from `u0`, indexed by problem variable.
`u0` is a vector, dictionary, or collection of `index => value` pairs. Pass existing
Lux parameters and state explicitly. `variables` selects original state indices;
fixed and pass-through initial conditions are skipped. Return the same `ps`, with
its existing buffers updated in place. Bounds are validated before writing.
"""
function constant_initialization(
        rng::Random.AbstractRNG, layer::ShootingLayer, ps, st;
        u0 = layer.problem.u0, variables = Base.OneTo(length(layer.initialcondition.initialconditions))
    )
    entries = shooting_initialization_entries(layer, ps, st, variables)
    set_shooting_guesses!(layer, entries, (i, t, j) -> initialization_value(u0, i))
    return ps
end

"""
$(SIGNATURES)

Omit `ps` and `st` to run `LuxCore.setup(rng, layer)` and return initialized `(ps, st)`.

Set tunable shooting values from `u0s`, a dictionary (or indexable collection)
mapping original variable indices to vectors in that variable's reset-grid index
order. `variables` defaults to dictionary keys, or all variables for other
collections. Update existing `ps` buffers and return `ps`; fixed values and `st`
are unchanged. All selected vectors must match their current schedules.
"""
function custom_initialization(
        rng::Random.AbstractRNG, layer::ShootingLayer, ps, st;
        u0s, variables = u0s isa AbstractDict ? keys(u0s) : Base.OneTo(length(layer.initialcondition.initialconditions))
    )
    entries = shooting_initialization_entries(layer, ps, st, variables)
    for entry in entries
        length(initialization_value(u0s, entry.index)) == length(entry.values) ||
            throw(DimensionMismatch("Expected $(length(entry.values)) guesses for variable $(entry.index)."))
    end
    set_shooting_guesses!(layer, entries, (i, t, j) -> initialization_value(u0s, i)[j])
    return ps
end

"""
$(SIGNATURES)

Omit `ps` and `st` to run `LuxCore.setup(rng, layer)` and return initialized `(ps, st)`.

Interpolate from `u0` (default: the problem initial state) to the required
`u_infinity` over the problem time span. Sample each variable's current reset grid
from `st`. Endpoints are vectors or dictionaries indexed by original variable.
Update tunable `ps` buffers and return `ps`. `variables` selects state indices;
fixed values are unchanged. Bounds are validated before writing.
"""
function linear_initialization(
        rng::Random.AbstractRNG, layer::ShootingLayer, ps, st;
        u_infinity, u0 = layer.problem.u0,
        variables = Base.OneTo(length(layer.initialcondition.initialconditions))
    )
    entries = shooting_initialization_entries(layer, ps, st, variables)
    t0, tf = layer.problem.tspan
    tf > t0 || throw(ArgumentError("Linear initialization requires an increasing, nonempty time span."))
    set_shooting_guesses!(
        layer, entries, (i, t, j) -> begin
            start = initialization_value(u0, i)
            start + (initialization_value(u_infinity, i) - start) * (t - t0) / (tf - t0)
        end
    )
    return ps
end

"""
$(SIGNATURES)

Omit `ps` and `st` to run `LuxCore.setup(rng, layer)` and return initialized `(ps, st)`.

Sample each selected tunable shooting value uniformly within its variable bounds
using `rng`. All selected bounds must be finite. `variables` selects original
state indices. Update existing buffers and return `ps`; fixed values and `st` are
unchanged. Use a seeded RNG for reproducible guesses.
"""
function random_initialization(
        rng::Random.AbstractRNG, layer::ShootingLayer, ps, st;
        variables = Base.OneTo(length(layer.initialcondition.initialconditions))
    )
    entries = shooting_initialization_entries(layer, ps, st, variables)
    bounds = Dict(entry.index => shooting_initialization_bounds(layer, entry.index) for entry in entries)
    all(b -> all(isfinite, b), values(bounds)) ||
        throw(ArgumentError("Random initialization requires finite lower and upper bounds."))
    set_shooting_guesses!(
        layer, entries, (i, t, j) -> begin
            lo, hi = bounds[i]
            lo + rand(rng) * (hi - lo)
        end
    )
    return ps
end

"""
$(SIGNATURES)

Omit `ps` and `st` to run `LuxCore.setup(rng, layer)` and return initialized `(ps, st)`.

Initialize selected interior shooting values by a sequential solve using the
current parameter schedule. Preserve assignments at the global start, fixed
resets, and unselected tunable resets. At each selected reset, use the incoming
solution value, including resets at the final time. Selected matching residuals
then vanish up to solver accuracy.

Pass existing `ps` and `st`. `variables` selects original state indices. Work on
private copies while solving, then validate bounds and update the existing `ps`
buffers. Return `ps`; the supplied state and fixed values remain unchanged. This
helper currently supports ODE problems and uses the layer's solver serially,
regardless of its ensemble algorithm.
"""
function forward_initialization(
        rng::Random.AbstractRNG, layer::ShootingLayer, ps, st;
        variables = Base.OneTo(length(layer.initialcondition.initialconditions))
    )
    layer.problem isa SciMLBase.AbstractODEProblem ||
        throw(ArgumentError("Forward initialization currently supports ODE problems."))
    entries = shooting_initialization_entries(layer, ps, st, variables)
    isempty(entries) && return ps
    work_ps, work_st = deepcopy(ps), deepcopy(st)
    work_entries = shooting_initialization_entries(layer, work_ps, work_st, variables)
    prob = layer.problem
    times = get_timestops(layer, ps, st)
    incoming = prob.u0
    for (k, t) in enumerate(times)
        if t != first(prob.tspan)
            for entry in work_entries
                haskey(entry.grid, t) && (entry.values[entry.grid[t]] = incoming[entry.index])
            end
        end
        (u0, _), ic_st = layer.initialcondition((incoming, t), work_ps.initialcondition, work_st.initialcondition; matching = false)
        p, p_st = layer.parameter(t, work_ps.parameter, work_st.parameter)
        work_st = merge(work_st, (; initialcondition = ic_st, parameter = p_st))
        k == length(times) && break
        sol = solve(
            remake(prob; u0, p, tspan = (t, times[k + 1])), layer.alg;
            save_everystep = false, dense = false, save_start = false, save_end = true
        )
        SciMLBase.successful_retcode(sol) || error("Forward initialization failed: $(sol.retcode)")
        incoming = terminal_state(sol)
    end
    guesses = Dict(entry.index => entry.values for entry in work_entries)
    set_shooting_guesses!(layer, entries, (i, t, j) -> guesses[i][j])
    return ps
end

"""
$(SIGNATURES)

Omit `ps` and `st` to run `LuxCore.setup(rng, layer)` and return initialized `(ps, st)`.

Apply `variables => helper` pairs sequentially to existing shooting parameters.
Selectors are original state indices or collections of indices. Each helper is
called as `helper(rng, layer, ps, st; variables, kwargs...)`. Use closures to supply
strategy-specific keywords when mixing methods. Later pairs override earlier
assignments. Return the same `ps`. Fixed values are never selected by the built-in
helpers. Failure of a later helper does not undo earlier successful updates.
"""
function hybrid_initialization(
        rng::Random.AbstractRNG, layer::ShootingLayer, ps, st, methods::Pair...;
        kwargs...
    )
    for (variables, method) in methods
        method(rng, layer, ps, st; variables, kwargs...)
    end
    return ps
end

# Convenience entry points perform setup once and retain its state for solving.
for initfun in (
        :forward_initialization, :linear_initialization, :random_initialization,
        :constant_initialization, :custom_initialization,
    )
    @eval function $initfun(rng::Random.AbstractRNG, layer::ShootingLayer; kwargs...)
        ps, st = LuxCore.setup(rng, layer)
        $initfun(rng, layer, ps, st; kwargs...)
        return ps, st
    end
end

function hybrid_initialization(
        rng::Random.AbstractRNG, layer::ShootingLayer, methods::Pair...; kwargs...
    )
    ps, st = LuxCore.setup(rng, layer)
    hybrid_initialization(rng, layer, ps, st, methods...; kwargs...)
    return ps, st
end
