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
configure function/derivative options. Initialization and layout preparation are
not repeated.
The `vectorize` keyword selects the packing implementation. Its default is
[`default_vectorizer`](@ref), which requires a loaded supported extension. Other
implementations may be supplied explicitly through `vectorize`.
"""
function SciMLBase.OptimizationFunction(
        prob::CorleoneDynamicOptProblem, ad::SciMLBase.ADTypes.AbstractADType;
        vectorize = default_vectorizer(),
        ps = prob.ps, inplace::Bool = true, kwargs...
    )
    _, reconstruct = pack(vectorize, ps)
    return optimization_function(prob, ad, reconstruct; inplace, kwargs...)
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
configures OptimizationFunction; remaining keywords are forwarded to
OptimizationProblem.
"""
function SciMLBase.OptimizationProblem(
        prob::CorleoneDynamicOptProblem, ad::SciMLBase.ADTypes.AbstractADType;
        vectorize = default_vectorizer(),
        ps = prob.ps, st = prob.st, inplace::Bool = true,
        function_kwargs::NamedTuple = (;), kwargs...
    )
    initial, reconstruct = pack(vectorize, ps)
    lower, upper = get_bounds(prob.layer, ps, st)
    lb, _ = pack(vectorize, lower)
    ub, _ = pack(vectorize, upper)
    axes(lb) == axes(ub) == axes(initial) ||
        throw(ArgumentError("Packed parameter and bound axes must match."))
    f = optimization_function(prob, ad, reconstruct; inplace, function_kwargs...)
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
