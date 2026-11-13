"""
    pack(extension::Val, ps) -> q0, reconstruct

Prepare flat real optimizer storage and a callable restoring named parameter access.
`reconstruct(q)` must use the current vector's numerical values and scalar type;
its result may be a ComponentArray, NamedTuple, or another parameter container.
Each implementation must pack parameters and their bounds in the same order.

Load ComponentArrays for `Val(:ComponentArrays)`. It returns the ComponentVector's
backing array and a callable retaining only its axes. Calling
that object wraps any compatible vector in a ComponentArray without copying numeric
values. Packing a NamedTuple allocates flat storage once; packing an existing
ComponentVector reuses its backing storage.
"""
function pack end

"""
$(SIGNATURES)

Select a loaded parameter vectorization extension. Currently returns
`Val(:ComponentArrays)` when the ComponentArrays extension is loaded. If no
supported extension is loaded, throw `UndefKeywordError(:vectorize)`.

Load ComponentArrays to enable the default, or supply `vectorize` explicitly to
OptimizationFunction/OptimizationProblem using an implementation of [`pack`](@ref).
"""
function default_vectorizer()
    if !isnothing(Base.get_extension(@__MODULE__, :CorleoneComponentArraysExtension))
        return Val(:ComponentArrays)
    end
    throw(UndefKeywordError(:vectorize))
end

"""
$(SIGNATURES)

Build typed optimizer callbacks using the reconstruction callable returned by [`pack`](@ref).
`inplace = true` uses the buffer-writing constraint callback; `false` uses the
allocating form and out-of-place derivative convention. Additional keywords
configure function/derivative options. `sparse_constraints = true` prepares a
colored sparse constraint Jacobian when DifferentiationInterface and
SparseMatrixColorings are loaded. `sparse_hessian = true` prepares colored
objective and Lagrangian Hessian callbacks. Both require `inplace = true`; the
Hessian option additionally requires second-order support from `ad`.
Initialization and layout preparation are not repeated.
The `vectorize` keyword selects the packing implementation. Its default is
[`default_vectorizer`](@ref), which requires a loaded supported extension. Other
implementations may be supplied explicitly through `vectorize`.
"""
function SciMLBase.OptimizationFunction(
        prob::CorleoneDynamicOptProblem, ad::SciMLBase.ADTypes.AbstractADType;
        vectorize = default_vectorizer(),
        ps = prob.ps, st = prob.st, inplace::Bool = true, sparse_constraints::Bool = false, sparse_hessian::Bool = false, kwargs...
    )
    initial, reconstruct = pack(vectorize, ps)
    sparse_constraints && !inplace && throw(ArgumentError("sparse_constraints requires inplace=true."))
    check_sparse_constraint_keywords(sparse_constraints, kwargs)
    sparse_hessian && !inplace && throw(ArgumentError("sparse_hessian requires inplace=true."))
    check_sparse_hessian_keywords(sparse_hessian, kwargs)
    jacobian_options = sparse_constraints ? sparse_constraint_options(prob, ad, initial, reconstruct, st) : (;)
    hessian_options = sparse_hessian ? sparse_hessian_options(prob, ad, initial, reconstruct, st) : (;)
    options = merge(jacobian_options, hessian_options)
    return optimization_function(prob, ad, reconstruct; st, inplace, options..., kwargs...)
end

function optimization_function(prob, ad, reconstruct; st = prob.st, inplace::Bool = true, kwargs...)
    haskey(kwargs, :cons) && throw(ArgumentError("Declare constraints on CorleoneDynamicOptProblem; its generated constraint callback cannot be replaced."))
    objective = (q, state) -> optimization_objective(prob, reconstruct(q), isnothing(state) ? st : state)
    constraints = if isempty(prob.lcons)
        nothing
    elseif inplace
        (res, q, state) -> optimization_constraints!(res, prob, reconstruct(q), isnothing(state) ? st : state)
    else
        (q, state) -> optimization_constraints(prob, reconstruct(q), isnothing(state) ? st : state)
    end
    return SciMLBase.OptimizationFunction{inplace}(objective, ad; cons = constraints, kwargs...)
end

"""
$(SIGNATURES)

Build a SciMLBase OptimizationProblem with `vectorize = default_vectorizer()`,
flattened Lux parameters, matching symbolic
bounds, and prepared user/shooting constraints. Use supplied `ps`/`st` or the model's
retained values. The packing implementation controls storage and reconstruction;
the default ComponentArrays implementation uses flat numerical storage and restores
ComponentArray axes around the current optimizer input without copying values. `function_kwargs`
configures OptimizationFunction. `sparse_constraints = true` prepares only the
constraint Jacobian sparsely; objective differentiation retains `ad`. Load
DifferentiationInterface and SparseMatrixColorings to enable it.
`sparse_hessian = true` additionally prepares exact objective and Lagrangian
Hessians with structural coloring; `ad` must support second differentiation.
Integer states and parameters (annotations with `integer = true`, or integral
symbols; see [`get_integrality`](@ref)) set the problem's `int` mask, one entry
per packed value. It is `nothing` when every value is continuous. Supplying `int`
explicitly replaces this mask. Remaining keywords are forwarded to
OptimizationProblem.
"""
function SciMLBase.OptimizationProblem(
        prob::CorleoneDynamicOptProblem, ad::SciMLBase.ADTypes.AbstractADType;
        vectorize = default_vectorizer(),
        ps = prob.ps, st = prob.st, inplace::Bool = true, sparse_constraints::Bool = false, sparse_hessian::Bool = false,
        function_kwargs::NamedTuple = (;), int = nothing, kwargs...
    )
    initial, reconstruct = pack(vectorize, ps)
    lower, upper = get_bounds(prob.layer, ps, st)
    lb, _ = pack(vectorize, lower)
    ub, _ = pack(vectorize, upper)
    axes(lb) == axes(ub) == axes(initial) ||
        throw(ArgumentError("Packed parameter and bound axes must match."))
    if isnothing(int)
        integral, _ = pack(vectorize, get_integrality(prob.layer, ps, st))
        axes(integral) == axes(initial) || throw(ArgumentError("Packed parameter and integrality axes must match."))
        int = any(!iszero, integral) ? Bool.(collect(integral)) : nothing
    end
    check_sparse_constraint_keywords(sparse_constraints, function_kwargs)
    sparse_constraints && !inplace && throw(ArgumentError("sparse_constraints requires inplace=true."))
    sparse_hessian && !inplace && throw(ArgumentError("sparse_hessian requires inplace=true."))
    check_sparse_hessian_keywords(sparse_hessian, function_kwargs)
    jacobian_options = sparse_constraints ? sparse_constraint_options(prob, ad, initial, reconstruct, st) : (;)
    hessian_options = sparse_hessian ? sparse_hessian_options(prob, ad, initial, reconstruct, st) : (;)
    options = merge(jacobian_options, hessian_options)
    f = optimization_function(prob, ad, reconstruct; st, inplace, options..., function_kwargs...)
    lcons, ucons = isempty(prob.lcons) ? (nothing, nothing) : (prob.lcons, prob.ucons)
    return SciMLBase.OptimizationProblem(f, initial, st; lb, ub, lcons, ucons, int, kwargs...)
end

"""
$(SIGNATURES)

Prepare a CorleoneDynamicOptProblem and construct its SciMLBase OptimizationProblem
in one call. Setup runs only when neither `ps` nor `st` is supplied. `loss`,
`constraints`, and initialization options follow the core model constructor.
"""
function SciMLBase.OptimizationProblem(
        layer::ShootingLayer, ad::SciMLBase.ADTypes.AbstractADType;
        vectorize = default_vectorizer(),
        loss = nothing, constraints = (), ps = nothing, st = nothing,
        lcons = nothing, ucons = nothing, eval_module::Module = @__MODULE__,
        evaluator = default_evaluator(),
        rng::Random.AbstractRNG = Random.default_rng(), kwargs...
    )
    problem = CorleoneDynamicOptProblem(layer, loss, constraints...; ps, st, rng, lcons, ucons, eval_module, evaluator)
    return SciMLBase.OptimizationProblem(problem, ad; vectorize, kwargs...)
end

"""
$(SIGNATURES)

Return a conservative sparse Boolean constraint Jacobian pattern for a prepared
ODE shooting problem. Columns follow `pack(vectorize, ps)`; rows follow
[`optimization_constraints`](@ref). Parsed Expr constraints (including lowered MTK
constraints) use their referenced state/time and parameter/time dependencies.
Literal vectors retain separate dependencies per component. Elementwise series
expressions such as `:(identity(u₁))` or `:(u₁ .- 0.5 .* p₂)` (see
[`Corleone.elementwise_series_expression`](@ref)) give row `k` only the
dependencies of saved sample `k`, plus any timed signals. A saved sample depends on
the values starting its segment, the active parameters and the controls applied
before it; both sides of a saved reset are kept apart. Other vector-valued
expressions use their combined dependencies. Arbitrary callback rows are dense
over the parameters they can access. Shooting rows
propagate all state dependencies within each integration segment, replace them
at scheduled resets, and include active tunable parameters. Fixed resets erase
state dependencies; partial resets retain dependencies of the other states.
Inserted parameter equalities have two entries each.

The pattern describes a fixed schedule and parameter layout. Rebuild it after
changing either. This computes structural metadata without solving or tracing
through the ODE solver. Dependencies of fixed numerical values in `st` are
excluded because the Jacobian differentiates only `ps`.
"""
function constraint_jacobian_sparsity(
        @nospecialize(prob::CorleoneDynamicOptProblem); vectorize = default_vectorizer(),
        ps = prob.ps, st = prob.st
    )
    initial, reconstruct = pack(vectorize, ps)
    return constraint_sparsity(prob, initial, reconstruct, st)
end

function constraint_sparsity(@nospecialize(prob), initial, reconstruct, st)
    if prob.layer isa EvaluationLayer && isnothing(prob.layer.sparsity)
        return copy(prepare_evaluation_sparsity!(prob, initial, reconstruct, st).jacobian)
    end
    dependencies = optimization_dependency_rows(prob, initial, reconstruct, st)
    return constraint_pattern(dependencies, length(prob.lcons), length(initial))
end

function constraint_pattern(dependencies, nrows, ncolumns)
    return dependency_pattern(vcat(dependencies.user, dependencies.matching), nrows, ncolumns)
end

"""
    dependency_pattern(rows, nrows, ncolumns)

Return the `nrows × ncolumns` Boolean sparse matrix whose row `i` has entries at
the column ids in `rows[i]`, typically `BitSet`s produced by
[`optimization_dependency_rows`](@ref) or [`shooting_dependency_context`](@ref).
"""
function dependency_pattern(all_rows, nrows, ncolumns)
    rows, columns = Int[], Int[]
    entries = sum(length, all_rows; init = 0)
    sizehint!(rows, entries)
    sizehint!(columns, entries)
    for (i, columns_for_row) in enumerate(all_rows), column in columns_for_row
        push!(rows, i)
        push!(columns, column)
    end
    return sparse(rows, columns, trues(length(rows)), nrows, ncolumns)
end

"""
    dependency_column_ids(prob, reconstruct, ncolumns, st)

Return the parameter tree of `prob` with each leaf replaced by its packed column id,
`reconstruct(1:ncolumns)` normalized for repeated tree access. Throw `ArgumentError`
when the shooting schedule in `st` no longer matches the one `prob` was prepared for.
Pass the result, or its subtrees, as `ids` to [`shooting_dependency_context`](@ref).
"""
function dependency_column_ids(@nospecialize(prob), reconstruct, ncolumns, st)
    check_optimization_schedule(prob.source_timestops, optimization_schedule(prob.layer, st))
    return sparsity_indices(reconstruct(collect(1:ncolumns)))
end

"""
    optimization_dependency_rows(prob, initial, reconstruct, st; hessian = false)

Return the structural column dependencies of a prepared problem as
`(; user, matching, matching_hessian, objective)`, each a vector of `BitSet`s of
packed column ids. `user` has one row per user constraint entry and `matching` one
per shooting equality; with `hessian = true`, `matching_hessian` holds the
curvature rows of the shooting equalities and `objective` the objective's coupled
column cliques, otherwise both are empty. [`constraint_jacobian_sparsity`](@ref)
and [`lagrangian_hessian_sparsity`](@ref) build their patterns from these rows.

Sublibraries add methods for problems whose layer is not a shooting layer, building
the rows with [`dependency_column_ids`](@ref), [`shooting_dependency_context`](@ref)
and [`trajectory_constraint_dependencies`](@ref).
"""
# Shared setup-time signal resolution for Jacobian and Hessian patterns.
# Structural setup does not specialize on the problem type: each distinct set of
# compiled expressions would otherwise recompile this whole analysis.
function optimization_dependency_rows(@nospecialize(prob), initial, reconstruct, st; hessian = false)
    prob.layer isa Union{ShootingLayer, EvaluationLayer} || throw(ArgumentError("Structural derivatives currently require a shooting layer."))
    packed_ids = dependency_column_ids(prob, reconstruct, length(initial), st)
    prepared_state = optimization_state(prob.layer, st, prob.evaluation_grid)
    layer, ids, state = shooting_components(prob.layer, packed_ids, prepared_state)
    layer.problem isa SciMLBase.AbstractODEProblem || throw(ArgumentError("Structural derivatives currently require an ODEProblem."))
    times = prob.times
    # Builtin field access keeps the heterogeneous constraint tuple from
    # specializing setup code (and its closures) on each expression type.
    constraints = prob.constraints
    functions = Any[getfield(getfield(constraints, i), :f) for i in 1:nfields(constraints)]
    lengths = Int[getfield(getfield(constraints, i), :n) for i in 1:nfields(constraints)]
    inspected = hessian ? vcat(functions, Any[prob.objective]) : functions
    dependencies, resolve = shooting_dependency_context(layer, ids, state, times, inspected; evaluation = prob.layer, packed_ids, hessian)
    length(dependencies.matching) == prob.nshooting || throw(ArgumentError("Shooting layout changed; reconstruct CorleoneDynamicOptProblem."))
    user_rows = BitSet[]
    for (f, n) in zip(functions, lengths)
        append!(user_rows, trajectory_constraint_dependencies(f, n, resolve, length(initial), length(times)))
    end
    objective = hessian ? trajectory_constraint_dependencies(prob.objective, 1, resolve, length(initial), length(times)) : BitSet[]
    hessian && append!(objective, evaluation_prior_columns(prob.layer, packed_ids))
    return (; user = user_rows, matching = dependencies.matching, matching_hessian = dependencies.matching_hessian, objective)
end

"""
    shooting_dependency_context(shooting, ids, st, times, expressions; evaluation = nothing, packed_ids = nothing, hessian = false, saved_samples = false)

Analyze which packed columns the trajectory of `shooting` depends on, without solving.
`ids` is the shooting parameter tree of column ids, `st` the prepared shooting state,
and `times` the saved time grid. `expressions` lists the constraint and objective
functions to inspect; parsed expressions refine which samples are resolved and other
values are ignored. Pass the wrapping `evaluation::EvaluationLayer` and the packed
`packed_ids` tree containing `ps.evaluation` ids to resolve evaluation signals.
Set `saved_samples = true` to resolve states at individual saved samples.

Return `(dependencies, resolve)`. `dependencies.matching` and
`dependencies.matching_hessian` hold the shooting-equality rows.
`resolve(signals, sample)` returns the `BitSet` of columns that the
`(kind, index, time)` signals read, where `sample` selects a saved sample index for
series reads, or `nothing` for their entire history. See [`expression_signals`](@ref).
"""
# Saved-index reads may distinguish the two sides of a reset. Include the
# history as a conservative fallback when the saved grid cannot be walked.
function shooting_dependency_context(
        layer, ids, state, times, inspected;
        evaluation = nothing, packed_ids = nothing, hessian = false, saved_samples = false
    )
    evaluation_ids = evaluation_coordinate_ids(evaluation, packed_ids)
    state_reads = saved_samples
    series_reads = saved_samples
    state_times = eltype(times)[]
    for f in inspected
        f isa ParsedTrajectoryExpression || continue
        for (kind, index, time) in f.dependencies
            kind === :states || continue
            state_reads = true
            isnothing(time) ? (series_reads |= f.elementwise) : push!(state_times, time)
        end
    end
    sort!(unique!(state_times))
    dependencies = shooting_constraint_dependencies(
        layer, ids, state; sample_times = state_reads ? state_times : nothing,
        series_times = series_reads ? times : nothing, hessian
    )
    parameter_layer = layer.parameter
    parameter_locations = parameter_signal_locations(parameter_layer, state.parameter)
    parameter_samples = Dict{Tuple{Int, Real}, BitSet}()
    parameter_series = Dict{Int, BitSet}()
    # Per-sample series reads need the saved-sample walk; otherwise use the history.
    state_samples = dependencies.state_samples
    resolve = (signals, sample) -> begin
        columns = BitSet()
        for (kind, index, time) in signals
            # A series read at one saved sample is a timed read at that sample.
            sampled = isnothing(time) && !isnothing(sample)
            time = sampled ? times[sample] : time
            kind === :time && continue
            if kind === :evaluation
                id = evaluation_ids[index]
                isnothing(id) || push!(columns, id)
            elseif kind === :states
                values = if sampled
                    isnothing(state_samples) ? dependencies.state_series : state_samples[sample]
                else
                    isnothing(time) ? dependencies.state_series : dependencies.states_at[time]
                end
                union!(columns, values[index])
            else
                dependency = if isnothing(time)
                    get!(parameter_series, index) do
                        # Bound the sampled interval indices by the first and
                        # last saved times without a lookup at every sample.
                        parameter_signal_dependency(parameter_layer, ids.parameter, state.parameter, parameter_locations[index], (first(times), last(times)))
                    end
                else
                    get!(parameter_samples, (index, time)) do
                        parameter_signal_dependency(parameter_layer, ids.parameter, state.parameter, parameter_locations[index], time)
                    end
                end
                union!(columns, dependency)
            end
        end
        columns
    end
    return dependencies, resolve
end

function check_sparse_constraint_keywords(enabled, kwargs)
    if enabled && any(key -> haskey(kwargs, key), (:cons_j, :cons_jac_prototype, :cons_jac_colorvec))
        throw(ArgumentError("sparse_constraints prepares cons_j and its sparsity metadata; do not also supply them."))
    end
    return nothing
end
sparse_constraint_options(args...) = throw(ArgumentError("Load DifferentiationInterface and SparseMatrixColorings to use sparse_constraints=true."))

# Vectorizers can normalize column IDs for repeated setup-time tree access.
# This hook only receives newly allocated integer metadata, never numerical ps.
sparsity_indices(ids) = ids

"""
$(SIGNATURES)

Return a conservative symmetric Boolean sparsity pattern for the Lagrangian
Hessian `σ * ∇²objective + sum(λ[i] * ∇²constraint[i])`, including the objective
when there are no constraints. Columns and rows follow `pack(vectorize, ps)`.
The pattern is independent of the current objective factor and multipliers.

Parsed expressions (including lowered MTK expressions) conservatively couple all
parameter columns on which each output depends. This covers general nonlinear
operations and curvature introduced by the ODE solution without differentiating
expression syntax. Unknown callbacks conservatively couple accessible columns.
This is a structural superset, not symbolic second differentiation.
State matching rows include curvature from the incoming solution and from
nonlinear transforms on assigned values. Parameter matching equalities are affine
and add no entries.

Supports prepared ODE shooting models, including experiment fits via CorleoneInverse.
Rebuild after schedule or layout
changes. Load OptimizationBase to obtain its canonical upper-triangle coordinates
with `OptimizationBase.lag_hess_structure(prob; vectorize, ps, st)`.
This helper supplies structure only. Use `sparse_hessian=true` when constructing
OptimizationFunction/OptimizationProblem to prepare numerical Hessian callbacks.
"""
function lagrangian_hessian_sparsity(
        @nospecialize(prob::CorleoneDynamicOptProblem); vectorize = default_vectorizer(),
        ps = prob.ps, st = prob.st
    )
    initial, reconstruct = pack(vectorize, ps)
    return lagrangian_sparsity(prob, initial, reconstruct, st)
end

function lagrangian_sparsity(@nospecialize(prob), initial, reconstruct, st)
    if prob.layer isa EvaluationLayer && isnothing(prob.layer.sparsity)
        return copy(prepare_evaluation_sparsity!(prob, initial, reconstruct, st).hessian)
    end
    dependencies = optimization_dependency_rows(prob, initial, reconstruct, st; hessian = true)
    return lagrangian_pattern(dependencies, length(initial))
end

function lagrangian_pattern(dependencies, ncolumns)
    cliques = unique(vcat(dependencies.objective, dependencies.user, dependencies.matching_hessian))
    columns = [BitSet() for _ in 1:ncolumns]
    for clique in cliques, column in clique
        union!(columns[column], clique)
    end
    colptr = Vector{Int}(undef, ncolumns + 1)
    rows = Int[]
    sizehint!(rows, sum(length, columns; init = 0))
    for (j, indices) in enumerate(columns)
        colptr[j] = length(rows) + 1
        append!(rows, indices)
    end
    colptr[end] = length(rows) + 1
    return SparseMatrixCSC(ncolumns, ncolumns, colptr, rows, fill(true, length(rows)))
end

function initialize_evaluation_sparsity!(prob)
    prob.layer.shooting.problem isa SciMLBase.AbstractODEProblem || return nothing
    isnothing(Base.get_extension(@__MODULE__, :CorleoneComponentArraysExtension)) && return nothing
    vectorize = default_vectorizer()
    applicable(pack, vectorize, prob.ps) || return nothing
    initial, reconstruct = pack(vectorize, prob.ps)
    prepare_evaluation_sparsity!(prob, initial, reconstruct, prob.st)
    return nothing
end

function prepare_evaluation_sparsity!(prob, initial, reconstruct, st)
    dependencies = optimization_dependency_rows(prob, initial, reconstruct, st; hessian = true)
    ncolumns = length(initial)
    patterns = EvaluationSparsity(
        constraint_pattern(dependencies, length(prob.lcons), ncolumns),
        lagrangian_pattern(dependencies, ncolumns)
    )
    prob.layer.sparsity = patterns
    return patterns
end

function check_sparse_hessian_keywords(enabled, kwargs)
    if enabled && any(key -> haskey(kwargs, key), (:hess, :hess_prototype, :hess_colorvec, :lag_h, :lag_hess_prototype, :lag_hess_colorvec))
        throw(ArgumentError("sparse_hessian prepares objective and Lagrangian Hessians and their metadata; do not also supply them."))
    end
    return nothing
end
sparse_hessian_options(args...) = throw(ArgumentError("Load DifferentiationInterface and SparseMatrixColorings to use sparse_hessian=true."))
