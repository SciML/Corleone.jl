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
SparseMatrixColorings are loaded; it requires `inplace = true`. Initialization and layout preparation are
not repeated.
The `vectorize` keyword selects the packing implementation. Its default is
[`default_vectorizer`](@ref), which requires a loaded supported extension. Other
implementations may be supplied explicitly through `vectorize`.
"""
function SciMLBase.OptimizationFunction(
        prob::CorleoneDynamicOptProblem, ad::SciMLBase.ADTypes.AbstractADType;
        vectorize = default_vectorizer(),
        ps = prob.ps, st = prob.st, inplace::Bool = true, sparse_constraints::Bool = false, kwargs...
    )
    initial, reconstruct = pack(vectorize, ps)
    sparse_constraints && !inplace && throw(ArgumentError("sparse_constraints requires inplace=true."))
    check_sparse_constraint_keywords(sparse_constraints, kwargs)
    options = sparse_constraints ? sparse_constraint_options(prob, ad, initial, reconstruct, st) : (;)
    return optimization_function(prob, ad, reconstruct; inplace, options..., kwargs...)
end

function optimization_function(prob, ad, reconstruct; inplace::Bool = true, kwargs...)
    haskey(kwargs, :cons) && throw(ArgumentError("Declare constraints on CorleoneDynamicOptProblem; its generated constraint callback cannot be replaced."))
    objective = (q, st) -> optimization_objective(prob, reconstruct(q), st)
    constraints = if isempty(prob.lcons)
        nothing
    elseif inplace
        (res, q, st) -> optimization_constraints!(res, prob, reconstruct(q), st)
    else
        (q, st) -> optimization_constraints(prob, reconstruct(q), st)
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
DifferentiationInterface and SparseMatrixColorings to enable it. Remaining keywords are forwarded to
OptimizationProblem.
"""
function SciMLBase.OptimizationProblem(
        prob::CorleoneDynamicOptProblem, ad::SciMLBase.ADTypes.AbstractADType;
        vectorize = default_vectorizer(),
        ps = prob.ps, st = prob.st, inplace::Bool = true, sparse_constraints::Bool = false,
        function_kwargs::NamedTuple = (;), kwargs...
    )
    initial, reconstruct = pack(vectorize, ps)
    lower, upper = get_bounds(prob.layer, ps, st)
    lb, _ = pack(vectorize, lower)
    ub, _ = pack(vectorize, upper)
    axes(lb) == axes(ub) == axes(initial) ||
        throw(ArgumentError("Packed parameter and bound axes must match."))
    check_sparse_constraint_keywords(sparse_constraints, function_kwargs)
    sparse_constraints && !inplace && throw(ArgumentError("sparse_constraints requires inplace=true."))
    options = sparse_constraints ? sparse_constraint_options(prob, ad, initial, reconstruct, st) : (;)
    f = optimization_function(prob, ad, reconstruct; inplace, options..., function_kwargs...)
    lcons, ucons = isempty(prob.lcons) ? (nothing, nothing) : (prob.lcons, prob.ucons)
    return SciMLBase.OptimizationProblem(f, initial, st; lb, ub, lcons, ucons, kwargs...)
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
Literal vectors retain separate dependencies per component; other vector-valued
expressions use their combined dependencies. Arbitrary callback rows are dense.
Shooting rows
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
        prob::CorleoneDynamicOptProblem; vectorize = default_vectorizer(),
        ps = prob.ps, st = prob.st
    )
    prob.layer isa ShootingLayer || throw(ArgumentError("Constraint sparsity currently requires a ShootingLayer."))
    initial, reconstruct = pack(vectorize, ps)
    return constraint_sparsity(prob, initial, reconstruct, st)
end

function constraint_sparsity(prob, initial, reconstruct, st)
    prob.layer isa ShootingLayer || throw(ArgumentError("Constraint sparsity currently requires a ShootingLayer."))
    check_optimization_schedule(prob.source_timestops, optimization_schedule(prob.layer, st))
    prob.layer.problem isa SciMLBase.AbstractODEProblem || throw(ArgumentError("Sparse shooting dependencies currently require an ODEProblem."))
    isempty(prob.lcons) && return spzeros(Bool, 0, length(initial))
    ids = sparsity_indices(reconstruct(collect(1:length(initial))))
    state = optimization_state(prob.layer, st, prob.evaluation_grid)
    state_reads = false
    state_times = eltype(prob.times)[]
    for constraint in prob.constraints
        constraint.f isa ParsedTrajectoryExpression || continue
        for (kind, index, time) in constraint.f.dependencies
            kind === :states || continue
            state_reads = true
            isnothing(time) || push!(state_times, time)
        end
    end
    sort!(unique!(state_times))
    dependencies = shooting_constraint_dependencies(prob.layer, ids, state; sample_times = state_reads ? state_times : nothing)
    length(dependencies.matching) == prob.nshooting || throw(ArgumentError("Shooting layout changed; reconstruct CorleoneDynamicOptProblem."))
    parameter_locations = parameter_signal_locations(prob.layer.parameter, state.parameter)
    parameter_samples = Dict{Tuple{Int, Real}, BitSet}()
    parameter_series = Dict{Int, BitSet}()
    resolve = signals -> begin
        columns = BitSet()
        for (kind, index, time) in signals
            if kind === :states
                values = isnothing(time) ? dependencies.state_series : dependencies.states_at[time]
                union!(columns, values[index])
            else
                dependency = if isnothing(time)
                    get!(parameter_series, index) do
                        # Bound the sampled interval indices by the first and
                        # last saved times without a lookup at every sample.
                        parameter_signal_dependency(prob.layer.parameter, ids.parameter, state.parameter, parameter_locations[index], (first(prob.times), last(prob.times)))
                    end
                else
                    get!(parameter_samples, (index, time)) do
                        parameter_signal_dependency(prob.layer.parameter, ids.parameter, state.parameter, parameter_locations[index], time)
                    end
                end
                union!(columns, dependency)
            end
        end
        columns
    end
    user_rows = BitSet[]
    for constraint in prob.constraints
        append!(user_rows, trajectory_constraint_dependencies(constraint.f, length(constraint), resolve, length(initial)))
    end
    all_rows = vcat(user_rows, dependencies.matching)
    rows, columns = Int[], Int[]
    entries = sum(length, all_rows; init = 0)
    sizehint!(rows, entries)
    sizehint!(columns, entries)
    for (i, columns_for_row) in enumerate(all_rows), column in columns_for_row
        push!(rows, i)
        push!(columns, column)
    end
    return sparse(rows, columns, trues(length(rows)), length(prob.lcons), length(initial))
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
