"""
    ParameterEstimationProblem <: SciMLBase.AbstractNonlinearProblem

A lightweight container tying a [`ParameterEstimationLayer`](@ref) to a flat vector of
optimization variables (the tunable parameters and initial conditions of the wrapped
shooting layer) so that the least-squares residuals can be minimized with any
NonlinearSolve solver.

For a plain parameter-estimation setup (no tunable initial conditions, no controls) the
optimization variables are exactly the tunable model parameters. Parameters that are fixed
on the wrapped layer (e.g. provided via `bounds_p` with coincident lower/upper bounds) are
not part of the optimization variables and are kept at their values.

With a multiple-shooting setup (a `ShootingLayer` with lifted local states) the flat vector
additionally contains the *local*
shooting initial conditions of every experiment (see [`ParameterEstimationLayer`](@ref)),
which keep the trajectory continuous at the shooting points.

# Parameter transforms

By default the optimizer works directly on the natural parameters. The `transforms` keyword
instead fits a *reparameterized* internal variable ``\\phi`` for every entry of the shared
`[u0; p]` block (local shooting states are never transformed), with the natural value
recovered as ``\\theta = \\mathrm{from\\_internal}(t, \\phi)`` (see
[`AbstractParameterTransform`](@ref)). A transform other than [`Identity`](@ref) replaces
the natural bounds of that variable with an unconstrained internal variable (for [`Log`](@ref)
and [`Logit`](@ref) the constraint is encoded in the transform's range). The flat vector
`x0` (and every other flat vector accepted or returned by this problem) is always expressed
in the *internal* space.

`transforms` accepts `nothing` (the default, all [`Identity`](@ref)), a single
[`AbstractParameterTransform`](@ref) applied to every shared entry, or a vector/tuple with
exactly one transform per shared entry (useful when tunable initial conditions and model
parameters need different transforms).

# Parameter regularization

By default the optimizer solves a pure (weighted) least-squares problem. The
`regularization` keyword augments the objective with a data-independent penalty on the
fitted *model parameters*, most commonly to stabilise ill-conditioned or
overparameterized fits. Regularization is expressed as extra residuals appended after the
observations (and any multiple-shooting continuity residuals), so `regularize` is honored by
every nonlinear least-squares solver (Gauss-Newton, Levenberg-Marquardt, trust-region, ...)
and by the Optimization.jl extension without any solver-specific code.

By default only the **model parameters** `p` of the shared `[u0; p]` block are penalized:
tuned *initial conditions* `u0` and the local multiple-shooting states are **never**
regularized. The penalty is evaluated on the **natural** parameter values (so `reference`
and per-parameter `weights` are in natural units). See [`AbstractRegularizer`](@ref),
[`L2Regularizer`](@ref), [`L1Regularizer`](@ref) and [`ElasticNetRegularizer`](@ref). Any
other callable mapping the penalized natural parameter vector to a residual vector can be
used directly as a custom regularizer.

# Fields
$(FIELDS)

# Constructors
- `ParameterEstimationProblem(layer; rng, x0, shooting_scale, transforms, regularization)`

The low-level `NonlinearSolve.NonlinearLeastSquaresProblem` can be built directly via
[`problem`](@ref). Most users construct that problem through the convenience constructor
`NonlinearSolve.NonlinearLeastSquaresProblem(pe::ParameterEstimationLayer; kwargs...)`.
"""
struct ParameterEstimationProblem{L, P, S, St, X, B, R, R2, TR, RG}
    "The wrapped parameter-estimation layer"
    layer::L
    "Structured initial parameters of the first experiment (`(u0, p, controls)` for single shooting, or an `(interval_1, ...)` NamedTuple for multiple shooting)"
    ps0::P
    "Tuple of the layer parameters of every experiment (used to expand the shared optimization variables per experiment for multiple shooting)"
    ps0s::S
    "The layer state"
    st::St
    "Flat vector of optimization variables (in the internal, transformed space)"
    x0::X
    "Flat lower bounds of the optimization variables (in the internal space, or `nothing`)"
    lb::B
    "Flat upper bounds of the optimization variables (in the internal space, or `nothing`)"
    ub::B
    "One [`AbstractParameterTransform`](@ref) per entry of the shared `[u0; p]` block (local shooting states are always `Identity`)"
    transforms::TR
    "In-place residual function `(res, x, p) -> res` reconstructing the structured layer parameters from `x`"
    residual::R
    "Out-of-place residual function `(x, p) -> r` (reverse-mode-AD compatible), see [`residual`](@ref)"
    residual_oop::R2
    "Multiplier applied to the appended shooting (continuity) residuals so they can be weighted relative to the observation residuals"
    shooting_scale::Float64
    "Total number of residuals (observations + multiple-shooting continuity constraints + regularization)"
    n_residuals::Int
    "Number of multiple-shooting continuity residuals (the `n_shooting_constraints` entries after the observation block)"
    n_shooting_constraints::Int
    "Number of regularization residuals (the trailing block), see [`AbstractRegularizer`](@ref)"
    n_regularization_residuals::Int
    "The resolved regularizer `(θ_pen) -> r_reg`, or `nothing` when no regularization is in effect"
    reg::RG
end

# Resolve the names of the tunable (selected) model parameters of `problem`'s first
# shooting layer, for display. Returns `nothing` when the names cannot be recovered (e.g.
# no symbol cache), in which case `show` falls back to a plain index list.
function _tunable_parameter_names(problem::ParameterEstimationProblem)
    layer = first(get_layer(problem.layer))
    names = Symbol[]
    for key in keys(layer.parameter.parameters)
        key in Corleone.control_symbols(layer.sys) && continue
        child = layer.parameter.parameters[key]
        if child isa Corleone.PiecewiseConstant && child.parameter isa NamedTuple
            for member in keys(child.parameter)
                child.parameter[member] isa Corleone.FixedLayer && continue
                push!(names, member)
            end
        elseif !(child isa Corleone.FixedLayer)
            push!(names, key)
        end
    end
    return isempty(names) ? nothing : String.(names)
end

# Engine shorthand used only for display purposes.
_transform_label(t) = string(nameof(typeof(t)))

function Base.show(io::IO, problem::ParameterEstimationProblem)
    type_color, no_color = SciMLBase.get_colorizers(io)
    pe = problem.layer
    n_exp = n_experiments(get_data(pe))
    shared = _shared_block(first(get_layer(pe)), problem.ps0)
    nu = length(shared.u0)
    np = length(shared.p)

    print(
        io, type_color, "ParameterEstimationProblem", no_color,
        _is_multiple_shooting(pe) ? " (multiple shooting, " : " (",
        type_color, n_exp, no_color, n_exp == 1 ? " experiment)" : " experiments)"
    )
    print(
        io, no_color, " with ", type_color, length(problem.x0), no_color,
        " optimization variable(s)\n"
    )

    # Fitted parameters, by name when resolvable, otherwise by count and index.
    names = _tunable_parameter_names(problem)
    print(io, no_color, "  fitted parameters (", type_color, np, no_color, "): ")
    if isnothing(names)
        print(io, join(_raw_param_indices(problem), ", "))
    else
        print(io, join(names, ", "))
    end
    nu > 0 && print(
        io, no_color,
        "\n  tunable initial conditions (", type_color, nu, no_color, ")"
    )

    # Residual composition: observations + shooting continuity + regularization.
    n_obs = problem.n_residuals - problem.n_shooting_constraints - problem.n_regularization_residuals
    print(io, no_color, "\n  residuals: ", type_color, n_obs, no_color, " observation(s)")
    if problem.n_shooting_constraints > 0
        print(
            io, no_color, " + ", type_color, problem.n_shooting_constraints, no_color,
            " shooting constraint(s)"
        )
    end
    if problem.n_regularization_residuals > 0
        print(
            io, no_color, " + ", type_color, problem.n_regularization_residuals, no_color,
            " regularization residual(s)"
        )
    end
    print(io, no_color, " = ", type_color, problem.n_residuals, no_color)

    # Transforms / regularization / shooting scale (only when non-trivial).
    has_t = any(t -> !(t isa Identity), problem.transforms)
    if has_t || !isnothing(problem.reg)
        print(io, no_color, "\n  ")
        if has_t
            print(io, "parameter transforms: ", join(_transform_label.(problem.transforms), ", "))
        end
        if !isnothing(problem.reg)
            has_t && print(io, "; ")
            print(io, nameof(typeof(problem.reg)), " regularization")
        end
    end
    problem.shooting_scale != 1.0 && print(
        io, no_color,
        "\n  shooting_scale: ", problem.shooting_scale
    )
    return nothing
end

# Raw indices (into the canonical `prob.p` vector) of the tunable model parameters, for
# the `show` fallback when names cannot be resolved. Returns the layers' `tunable_p`.
function _raw_param_indices(problem::ParameterEstimationProblem)
    layer = first(get_layer(problem.layer))
    return collect(Iterators.flatten(layer.parameter.indices))
end

# --------------------------------------------------------------------------------------
# Optimization-variable layout
# --------------------------------------------------------------------------------------
#
# The flat vector `x` of optimization variables is built from the layer parameters of each
# experiment, dispatching on the kind of shooting layer:
#
# * **Single shooting**: `ps = (u0, p, controls)` is shared between all experiments, and
#   `x = [u0; p]` -- the tunable initial conditions followed by the tunable model
#   parameters.
# * **Multiple shooting**: `ps = (interval_1 = ..., ..., interval_K = ...)`, one block per
#   shooting interval, and `x = [u0; p; λ_1; ...; λ_E]` -- the shared tunable initial
#   conditions `u0`, the shared tunable model parameters `p` (interval 1), and then, per
#   experiment `e`, the *local* shooting initial conditions `λ_e` of its intervals `2..K`.
#
# The `[u0; p]` block is the *shared model* block that [`estimate_parameters`](@ref)
# reports; the local shooting states `λ_e` are auxiliary continuity variables used only to
# keep the trajectory continuous at the shooting points.

"True if `pe` has any interior shooting state (i.e. was built from shooting points)."
function _is_multiple_shooting(pe::ParameterEstimationLayer)
    layers = get_layer(pe)
    for layer in layers
        for child in values(layer.initialcondition.initialconditions)
            child isa Corleone.InitialCondition || continue
            length(child.lookup.collection) > 1 && return true
        end
    end
    return false
end

# Parameter groups of `layer` that are *not* control slots (i.e. ordinary model parameters).
# Control slots are held fixed by the trajectory and are never optimization variables.
function _ordinary_parameter_keys(layer)
    controls = Corleone.control_symbols(layer.sys)
    return [k for k in keys(layer.parameter.parameters) if !(k in controls)]
end

# The trainable leaf value vectors of one (possibly grouped) parameter entry, in canonical
# order. A `FixedLayer` group (all entries fixed) contributes nothing.
function _parameter_leaves(layer_child, ps_child)
    if layer_child isa Corleone.FixedLayer
        return ()
    elseif layer_child isa Corleone.PiecewiseConstant && layer_child.parameter isa NamedTuple
        members = keys(layer_child.parameter)
        return tuple(
            (
                ps_child.parameter[m].ps for m in members
                    if !(layer_child.parameter[m] isa Corleone.FixedLayer)
            )...
        )
    else
        return (ps_child.parameter.ps,)
    end
end

# Concatenate the scalar values collected in `parts` (a collection of tuples of scalars),
# in order. Always returns a `Vector`, even for zero or one value.
function _collect_scalars(parts)
    values = [v for part in parts for v in part]
    return isempty(values) ? Float64[] : values
end

# Concatenate the value vectors collected in `parts` (a collection of tuples of vectors),
# in order. Always returns a `Vector`, even for zero or one non-empty vector.
function _collect_vectors(parts)
    nonempty = [v for part in parts for v in part if !isempty(v)]
    return isempty(nonempty) ? Float64[] : reduce(vcat, nonempty)
end

# Positions of the *shared* (global initial) value and of every interior (local shooting)
# value within an initial-condition grid. The shared position is the one at the global
# start; a `PassThrough` group has neither.
function _ic_positions(layer_child, t0)
    collection = layer_child.lookup.collection
    n = length(collection)
    sharedpos = get(collection, t0, nothing)
    # Typed comprehension: on Julia 1.10 the untyped filter comprehension is inferred as
    # eltype-unknown, which makes the returned `localpositions` vector abstract and
    # pollutes the element types of everything built from it downstream.
    localpositions = Int[i for i in 1:n if !isequal(i, sharedpos)]
    return sharedpos, localpositions
end

# The number of trainable state values of one initial-condition group (one per member).
function _ic_trainable_count(layer_child)
    if layer_child.parameter isa NamedTuple
        return count(m -> !(layer_child.parameter[m] isa Corleone.FixedLayer), keys(layer_child.parameter))
    end
    return 1
end

# The shared `(u0, p)` model block of the (first) layer parameters `ps0`. `u0` holds the
# global-start value of every tunable initial condition, `p` every trainable model
# parameter (controls excluded).
function _shared_block(layer, ps0::NamedTuple)
    t0 = first(layer.problem.tspan)
    ic_layers = layer.initialcondition.initialconditions
    ic_ps = ps0.initialcondition.initialconditions
    ic_parts = map(collect(keys(ic_layers))) do key
        child = ic_layers[key]
        child isa Corleone.InitialCondition || return ()
        sharedpos, _ = _ic_positions(child, t0)
        isnothing(sharedpos) && return ()
        leaf = ic_ps[key].parameter
        if child.parameter isa NamedTuple
            return tuple(
                (
                    leaf[m].ps[sharedpos] for m in keys(child.parameter)
                        if !(child.parameter[m] isa Corleone.FixedLayer)
                )...
            )
        end
        return (leaf.ps[sharedpos],)
    end
    p_parts = map(_ordinary_parameter_keys(layer)) do key
        _parameter_leaves(layer.parameter.parameters[key], ps0.parameter.parameters[key])
    end
    return (u0 = _collect_scalars(ic_parts), p = _collect_vectors(p_parts))
end

# The local shooting initial conditions of one experiment, i.e. the concatenation of every
# interior-grid value of its tunable initial conditions. Empty for a single-shooting layer.
function _local_states(layer, ps0_i::NamedTuple)
    t0 = first(layer.problem.tspan)
    ic_layers = layer.initialcondition.initialconditions
    ic_ps = ps0_i.initialcondition.initialconditions
    parts = map(collect(keys(ic_layers))) do key
        child = ic_layers[key]
        child isa Corleone.InitialCondition || return ()
        _, localpositions = _ic_positions(child, t0)
        isempty(localpositions) && return ()
        leaf = ic_ps[key].parameter
        if child.parameter isa NamedTuple
            return tuple(
                (
                    leaf[m].ps[localpositions] for m in keys(child.parameter)
                        if !(child.parameter[m] isa Corleone.FixedLayer)
                )...
            )
        end
        return (leaf.ps[localpositions],)
    end
    return _collect_vectors(parts)
end

"Flat vector of the optimization variables for the layer parameters `ps0s`."
function _flatten(pe::ParameterEstimationLayer, ps0s::Tuple)
    layers = get_layer(pe)
    shared = _shared_block(layers[1], ps0s[1])
    x = vcat(shared.u0, shared.p)   # [u0; p]
    for i in eachindex(ps0s)
        x = vcat(x, _local_states(layers[i], ps0s[i]))
    end
    return x
end

# Number of entries of `x` belonging to the shared `[u0; p]` model block.
function _n_shared(layer::ParameterEstimationLayer, ps0s::Tuple)
    shared = _shared_block(first(get_layer(layer)), ps0s[1])
    return length(shared.u0) + length(shared.p)
end

# Map the shared `[u0; p]` block of a flat *internal* vector `x` back to natural parameter
# values using `transforms` (one transform per shared entry). The remaining entries (the
# multiple-shooting local states) are never transformed.
function _natural_shared(x::AbstractVector, transforms::Tuple, nu::Int, np::Int)
    u0 = [from_internal(transforms[i], x[i]) for i in 1:nu]
    p = [from_internal(transforms[nu + j], x[nu + j]) for j in 1:np]
    return u0, p
end

# Map the shared `[u0; p]` block of a flat *natural* vector to internal values, leaving the
# local-state entries (beyond the first `m`) untouched. The default initial parameter values
# must lie in the domain of their transform; a non-finite or out-of-domain start is reported
# as a clear error rather than an obscure `DomainError`.
function _to_internal_shared(natural::AbstractVector, transforms::Tuple, m::Int)
    shared = map(1:m) do i
        θ = natural[i]
        φ = try
            to_internal(transforms[i], θ)
        catch
            throw(
                ArgumentError(
                    "The initial value $(θ) of shared optimization variable $i is not in the " *
                        "domain of transform $(transforms[i]). Provide a feasible initial parameter " *
                        "value or an explicit `x0` (in the internal space)."
                )
            )
        end
        isfinite(φ) || throw(
            ArgumentError(
                "The initial value $(θ) of shared optimization variable $i maps to the " *
                    "non-finite internal value $φ under transform $(transforms[i])."
            )
        )
        return φ
    end
    return vcat(shared, natural[(m + 1):end])
end

# Number of local shooting-state entries of one experiment (0 for single shooting).
_n_local_states(layer, ps0_i::NamedTuple) = length(_local_states(layer, ps0_i))

# Rebuild a named tuple from its compile-time `names` type parameter so the result keeps the
# concrete `NamedTuple{names}` type (a runtime `keys(nt)` value would only be known as
# `NTuple{N,Symbol}` and make the rebuild type-unstable).
_nt_rebuild(f::F, ::NamedTuple{names}) where {F, names} = NamedTuple{names}(f(names))

# Assemble one initial-condition value vector of length `n` from the shared value `uval`
# (placed at `sharedpos`) and the interior values `lvals` (placed at `localpositions`).
function _assemble_ic_vector(n::Int, sharedpos, localpositions, uval, lvals)
    return map(1:n) do i
        # `i` is either the shared position or one of the local positions, so `findfirst`
        # always succeeds; on Julia 1.10 the untyped `Union{Nothing, Int}` result would
        # widen this map's element type to `Any`, so assert the concrete index type.
        (!isnothing(sharedpos) && i == sharedpos) ? uval :
            lvals[findfirst(==(i), localpositions)::Int]
    end
end

# The shared-block/local-state extraction and scalar/vector gathering only inspect the
# incoming (constant) parameter buffers to derive metadata and base values. They must not be
# traced by reverse-mode AD, both for performance and because their assembly is mutating.
ChainRulesCore.@non_differentiable _collect_scalars(::Any)
ChainRulesCore.@non_differentiable _collect_vectors(::Any)
ChainRulesCore.@non_differentiable _shared_block(::Any, ::Any)
ChainRulesCore.@non_differentiable _local_states(::Any, ::Any)

# Rebuild the value vectors of one (possibly grouped) initial-condition entry. Dispatch on
# the *concrete* `ps_child.parameter` shape: a scalar collection stores its vector under the
# `:ps` key, while a grouped collection stores one child per variable name. The layer's own
# `parameter` field is sometimes only known as an abstract `ParameterCollection` type, so it
# cannot drive dispatch without losing type stability.
function _rebuild_ic_group(layer_child, ps_child, sharedpos, localpositions, uvals, lvals)
    return _rebuild_ic_group_ps(
        ps_child.parameter, layer_child, ps_child, sharedpos, localpositions, uvals, lvals
    )
end

function _rebuild_ic_group_ps(
        ::NamedTuple{(:ps,)}, layer_child, ps_child, sharedpos, localpositions, uvals, lvals
    )
    n = length(ps_child.parameter.ps)
    uval = isnothing(sharedpos) ? zero(eltype(lvals)) : only(uvals)
    v = _assemble_ic_vector(n, sharedpos, localpositions, uval, lvals)
    return merge(ps_child, (parameter = merge(ps_child.parameter, (ps = v,)),))
end

function _rebuild_ic_group_ps(
        ::NamedTuple{names}, layer_child, ps_child, sharedpos, localpositions, uvals, lvals
    ) where {names}
    newmembers = _nt_rebuild(ps_child.parameter) do members
        trainable = [m for m in members if !(layer_child.parameter[m] isa Corleone.FixedLayer)]
        nl = length(localpositions)
        map(members) do member
            j = findfirst(==(member), trainable)
            if isnothing(j)
                ps_child.parameter[member]
            else
                n = length(ps_child.parameter[member].ps)
                uval = isnothing(sharedpos) ? zero(eltype(lvals)) : uvals[j]
                localvals = lvals[((j - 1) * nl + 1):(j * nl)]
                v = _assemble_ic_vector(n, sharedpos, localpositions, uval, localvals)
                merge(ps_child.parameter[member], (ps = v,))
            end
        end
    end
    return merge(ps_child, (parameter = newmembers,))
end

# Rebuild the value vector of one (possibly grouped) parameter entry.
function _rebuild_param_group(layer_child, ps_child, values)
    if layer_child isa Corleone.FixedLayer
        return ps_child
    elseif layer_child isa Corleone.PiecewiseConstant && layer_child.parameter isa NamedTuple
        newmembers = _nt_rebuild(layer_child.parameter) do members
            trainable = [m for m in members if !(layer_child.parameter[m] isa Corleone.FixedLayer)]
            sizes = [length(ps_child.parameter[m].ps) for m in trainable]
            offsets = [1 + sum(sizes[1:(j - 1)]) for j in eachindex(trainable)]
            map(members) do member
                j = findfirst(==(member), trainable)
                isnothing(j) ? ps_child.parameter[member] :
                    merge(ps_child.parameter[member], (ps = values[offsets[j]:(offsets[j] + sizes[j] - 1)],))
            end
        end
        return merge(ps_child, (parameter = newmembers,))
    end
    return merge(ps_child, (parameter = merge(ps_child.parameter, (ps = values,)),))
end

# Rebuild the initial-condition container of one experiment from the shared `u0` block and
# the experiment's local shooting states `λ`.
# Prefix offsets of a tuple of block sizes: `(1, 1+n1, 1+n1+n2, ...)`. Recursing on the
# tuple keeps both the offsets and their element types concrete.
_prefix_offsets(sizes::Tuple) = _prefix_offsets(sizes, 1)
_prefix_offsets(::Tuple{}, ::Int) = ()
function _prefix_offsets(sizes::Tuple, offset::Int)
    return (offset, _prefix_offsets(Base.tail(sizes), offset + first(sizes))...)
end

# Rebuild the initial-condition container of one experiment from the shared `u0` block and
# the experiment's local shooting states `λ`.
function _rebuild_ic_tree(layer, ps, u0, λ)
    t0 = first(layer.problem.tspan)
    psnt = ps.initialcondition.initialconditions
    laynt = layer.initialcondition.initialconditions
    iclayers = values(laynt)
    pschildren = values(psnt)
    sharedsizes = map(iclayers) do c
        c isa Corleone.InitialCondition || return 0
        sharedpos, _ = _ic_positions(c, t0)
        isnothing(sharedpos) ? 0 : _ic_trainable_count(c)
    end
    localsizes = map(iclayers) do c
        c isa Corleone.InitialCondition || return 0
        _, localpositions = _ic_positions(c, t0)
        length(localpositions) * _ic_trainable_count(c)
    end
    uoffsets = _prefix_offsets(sharedsizes)
    loffsets = _prefix_offsets(localsizes)
    newgroups = _nt_rebuild(psnt) do _
        map(iclayers, pschildren, sharedsizes, localsizes, uoffsets, loffsets) do child, ps_child, ss, ls, uo, lo
            child isa Corleone.InitialCondition || return ps_child
            sharedpos, localpositions = _ic_positions(child, t0)
            uvals = u0[uo:(uo + ss - 1)]
            lvals = λ[lo:(lo + ls - 1)]
            _rebuild_ic_group(child, ps_child, sharedpos, localpositions, uvals, lvals)
        end
    end
    return merge(ps.initialcondition, (initialconditions = newgroups,))
end

# Sum the lengths of the entries of `parts` (a tuple/collection of arrays) without the
# keyword `sum(f, itr; init)` form, whose `kwcall` Zygote cannot differentiate.
function _total_length(parts)
    n = 0
    for part in parts
        n += length(part)
    end
    return n
end

# Rebuild the parameter container of one experiment from the trainable parameter values.
function _rebuild_param_tree(layer, ps, p)
    keys_ = keys(ps.parameter.parameters)
    sizes = map(keys_) do key
        key in Corleone.control_symbols(layer.sys) && return 0
        _total_length(_parameter_leaves(layer.parameter.parameters[key], ps.parameter.parameters[key]))
    end
    offsets = _prefix_offsets(sizes)
    index = Ref(0)
    newgroups = _nt_rebuild(ps.parameter.parameters) do names
        map(names) do key
            index[] += 1
            k = index[]
            # Control slots are prescribed by the data/trajectory and are not optimization
            # variables; leave their (already populated) parameter buffers untouched.
            key in Corleone.control_symbols(layer.sys) && return ps.parameter.parameters[key]
            _rebuild_param_group(
                layer.parameter.parameters[key], ps.parameter.parameters[key],
                p[offsets[k]:(offsets[k] + sizes[k] - 1)]
            )
        end
    end
    return merge(ps.parameter, (parameters = newgroups,))
end

# Reconstruct the structured layer parameters `ps` used by the layer callable (and the
# residual evaluation) from the flat internal vector `x`. The shared `[u0; p]` block is
# mapped back to natural units with `transforms`; the remaining entries are the local
# shooting states of the experiments in order.
function _unflatten(
        pe::ParameterEstimationLayer, ps0s::NTuple{N, Any}, x::AbstractVector, transforms
    ) where {N}
    layers = get_layer(pe)
    n_exp = N
    shared = _shared_block(layers[1], ps0s[1])
    nu = length(shared.u0)
    np = length(shared.p)
    m = nu + np
    @assert length(x) >= m
    u0, p = _natural_shared(x, transforms, nu, np)
    ps = ntuple(Val(N)) do i
        li = _n_local_states(layers[i], ps0s[i])
        start = m + 1 + (i - 1) * li
        λi = x[start:(start + li - 1)]
        ic = _rebuild_ic_tree(layers[i], ps0s[i], u0, λi)
        par = _rebuild_param_tree(layers[i], ps0s[i], p)
        merge(ps0s[i], (initialcondition = ic, parameter = par))
    end
    return n_exp == 1 ? only(ps) : ps
end

# --- Transforms --------------------------------------------------------------------
# Normalize the user-facing `transforms` keyword into a `Tuple` with one transform per entry
# of the shared `[u0; p]` block (`m` entries). Accepts `nothing` (all `Identity`), a single
# transform (applied to every shared entry), or a vector/tuple of exactly `m` transforms.
# A `Tuple` is used so that even a *mixed* choice keeps a concrete element type per slot and
# the transform application stays type-stable (`Vector{AbstractParameterTransform}` would not).
function _resolve_transforms(transforms, m::Int)
    transforms === nothing && return ntuple(_ -> Identity(), m)
    transforms isa AbstractParameterTransform && return ntuple(_ -> transforms, m)
    if transforms isa Union{AbstractVector, Tuple}
        length(transforms) == m || throw(
            ArgumentError(
                "Got $(length(transforms)) transform(s) for the shared `[u0; p]` block of " *
                    "length $m. Provide exactly one transform per shared optimization variable."
            )
        )
        all(t -> t isa AbstractParameterTransform, transforms) || throw(
            ArgumentError(
                "Every entry of `transforms` must be an `AbstractParameterTransform`."
            )
        )
        return Tuple(transforms)
    end
    throw(
        ArgumentError(
            "`transforms` must be `nothing`, an `AbstractParameterTransform`, or a " *
                "vector/tuple of them (got $(typeof(transforms)))."
        )
    )
end

# Flat internal bounds implied by the transforms. The shared block uses each transform's
# internal bounds (natural bounds for `Identity`, unconstrained for the others); the local
# shooting states keep their natural bounds.
function _transformed_bounds(transforms::Tuple, lb, ub, m::Int)
    T = promote_type(eltype(lb), Float64)
    out_lb = Vector{T}(undef, length(lb))
    out_ub = Vector{T}(undef, length(ub))
    for i in 1:m
        out_lb[i], out_ub[i] = _internal_bounds(transforms[i], lb[i], ub[i])
    end
    for i in (m + 1):length(lb)
        out_lb[i] = lb[i]
        out_ub[i] = ub[i]
    end
    return out_lb, out_ub
end

# --- Bounds ------------------------------------------------------------------------
# Flat natural bounds of every optimization variable, in the same order as `_flatten`:
# the shared `[u0; p]` block followed by the local shooting states of every experiment.
# `Corleone.get_bounds` returns a tree shaped like the layer parameters, so the same
# flattening helpers apply.
function _tunable_bounds(pe::ParameterEstimationLayer, ps0s::Tuple)
    layers = get_layer(pe)
    lb, ub = Corleone.get_bounds(layers[1], ps0s[1])
    shared_lb = _shared_block(layers[1], lb)
    shared_ub = _shared_block(layers[1], ub)
    flat_lb = vcat(shared_lb.u0, shared_lb.p)
    flat_ub = vcat(shared_ub.u0, shared_ub.p)
    for i in eachindex(ps0s)
        lbi, ubi = Corleone.get_bounds(layers[i], ps0s[i])
        flat_lb = vcat(flat_lb, _local_states(layers[i], lbi))
        flat_ub = vcat(flat_ub, _local_states(layers[i], ubi))
    end
    return flat_lb, flat_ub
end

# --------------------------------------------------------------------------------------
# Problem construction
# --------------------------------------------------------------------------------------
function ParameterEstimationProblem(
        layer::ParameterEstimationLayer;
        rng::Random.AbstractRNG = Random.default_rng(),
        x0 = nothing,
        shooting_scale::Real = 1.0,
        transforms = nothing,
        regularization = nothing,
        regularization_reference = nothing,
        regularization_weights = nothing,
        regularized_indices = nothing,
    )
    # One structured parameter set per experiment (its shooting layer's initial
    # parameters), plus the tuple of per-experiment layer states. `ps0` (the first
    # experiment's parameters) is the shared model block used by the accessors.
    n_exp = n_experiments(get_data(layer))
    ps0s = ntuple(i -> LuxCore.initialparameters(rng, get_layer(layer)[i]), n_exp)
    st = ntuple(i -> LuxCore.initialstates(rng, get_layer(layer)[i]), n_exp)
    # One transform per shared `[u0; p]` entry. The default `x0` is built from the natural
    # initial parameters and mapped to internal space; an explicitly supplied `x0` is taken
    # to be already internal (matching the space the optimizer works in).
    m = _n_shared(layer, ps0s)
    transforms = _resolve_transforms(transforms, m)
    x0 = isnothing(x0) ? _to_internal_shared(_flatten(layer, ps0s), transforms, m) : x0
    lb, ub = _transformed_bounds(transforms, _tunable_bounds(layer, ps0s)..., m)

    # Parameter regularization. `nu` tunable initial-condition entries are followed by `np`
    # model parameters in the shared block. By default only the `np` model parameters are
    # penalized; `regularized_indices` may extend (or restrict) the penalized set to any
    # subset of the shared `[u0; p]` block, so tuned initial conditions can be regularized
    # too when the user has a prior on them. The `regularization_reference` /
    # `regularization_weights` keywords, when given, override those carried by the
    # regularizer itself (they are the most common per-fit tunables) and are aligned with
    # the penalized set.
    nu = length(_shared_block(first(get_layer(layer)), ps0s[1]).u0)
    np = m - nu
    regularization = _apply_reg_overrides(regularization, regularization_reference, regularization_weights)
    reg, pen, n_reg = _resolve_regularization(regularization, regularized_indices, nu, np)

    # Optional weighting of the shooting (continuity) residuals. Observations are
    # residualized against their noise scale `σ`, so they are `O(1)` at the noise floor,
    # while the appended shooting residuals are raw state/parameter differences with the
    # state scale. For multiple shooting the reconstruction is only meaningfully continuous
    # when the optimizer drives the shooting residuals to (nearly) zero. If the optimal
    # solution still has a large shooting residual, increase `shooting_scale` and re-solve:
    # this weighs the continuity residuals more heavily so the optimizer satisfies them.
    # The first `n_obs` residuals are the (measured) observations, the remaining
    # `n_shooting_constraints` are the appended shooting residuals. Both counts are computed
    # once here (reusing the already-constructed layer states, which is type-stable) and
    # stored as concrete `Int` fields so downstream code does not have to recompute them
    # from the layer (which would call the not-type-stable `initialstates`).
    n_obs::Int = n_observations(get_data(layer))
    n_shoot::Int = sum(
        i -> _layer_shooting_constraints(get_layer(layer)[i], ps0s[i], st[i]), 1:n_exp
    )
    n_res = n_obs + n_shoot + n_reg
    # The per-experiment segment-boundary grids (layer switching times extended with the
    # observation times) are constant for a given layer/state, but the layer would
    # re-assemble them on every forward call (i.e. every solver iteration). Compute them
    # once here and pass them through to the layer evaluation in the residual closures.
    tstops = ntuple(
        i -> _experiment_tstops(get_data(layer), get_layer(layer)[i], st[i], i), n_exp
    )
    # In-place residual: observation residuals, then (scaled) shooting residuals, then the
    # regularization residuals. `shooting_scale` only weights the shooting block.
    residual = let layer = layer, ps0s = ps0s, st = st, n_obs = n_obs, n_shoot = n_shoot,
            transforms = transforms, shooting_scale = Float64(shooting_scale),
            reg = reg, pen = pen, n_reg = n_reg, tstops = tstops
        (res, x, p) -> begin
            # `residuals!` writes the observation block into `res[1:n_obs]` and the
            # multiple-shooting block into `res[n_obs+1 : n_obs+n_shoot]` of the full buffer;
            # the trailing regularization slot is filled below.
            ps = _unflatten(layer, ps0s, x, transforms)
            residuals!(res, layer, ps, st, tstops)
            if shooting_scale != 1.0 && n_shoot > 0
                @views res[(n_obs + 1):(n_obs + n_shoot)] .*= shooting_scale
            end
            if n_reg > 0
                θ_pen = [from_internal(transforms[i], x[i]) for i in pen]
                r_reg = reg(θ_pen)
                copyto!(view(res, (n_obs + n_shoot + 1):n_res), r_reg)
            end
            return res
        end
    end
    # Out-of-place counterpart used by the solvers when reverse-mode AD is requested (see
    # `residual` in `residuals.jl`). It reuses the functional residual path and applies the
    # same `shooting_scale` weighting and regularization without mutating in place. All
    # three blocks are built functionally so it stays `Zygote`/`ReverseDiff`-differentiable.
    residual_oop = let layer = layer, ps0s = ps0s, st = st, n_obs = n_obs, n_shoot = n_shoot,
            transforms = transforms, shooting_scale = Float64(shooting_scale),
            reg = reg, pen = pen, n_reg = n_reg, tstops = tstops
        (x, p) -> begin
            ps = _unflatten(layer, ps0s, x, transforms)
            r = CorleoneInverse.residual(layer, ps, st, tstops)
            # Scale the shooting block (entries `n_obs+1 .. n_obs+n_shoot`).
            r = if shooting_scale == 1.0
                r
            else
                vcat(r[1:n_obs], r[(n_obs + 1):(n_obs + n_shoot)] .* shooting_scale)
            end
            if n_reg > 0
                θ_pen = [from_internal(transforms[i], x[i]) for i in pen]
                r = vcat(r, reg(θ_pen))
            end
            return r
        end
    end

    return ParameterEstimationProblem(
        layer, ps0s[1], ps0s, st, x0, lb, ub, transforms, residual, residual_oop,
        Float64(shooting_scale), n_res, n_shoot, n_reg, reg
    )
end

# Apply the optional `regularization_reference` / `regularization_weights` overrides to a
# regularizer (returning a new regularizer with the tuned vectors) while leaving `nothing`
# (no regularization) and plain custom functors untouched.
function _apply_reg_overrides(reg, reference, weights)
    if isnothing(reg) || (isnothing(reference) && isnothing(weights))
        return reg
    end
    if reg isa L2Regularizer
        return L2Regularizer(
            reg.λ; reference = isnothing(reference) ? reg.reference : reference,
            weights = isnothing(weights) ? reg.weights : weights
        )
    elseif reg isa L1Regularizer
        return L1Regularizer(
            reg.λ; reference = isnothing(reference) ? reg.reference : reference,
            weights = isnothing(weights) ? reg.weights : weights
        )
    elseif reg isa ElasticNetRegularizer
        return ElasticNetRegularizer(
            reg.l1.λ, reg.l2.λ;
            reference = isnothing(reference) ? reg.l1.reference : reference,
            weights = isnothing(weights) ? reg.l1.weights : weights
        )
    end
    throw(
        ArgumentError(
            "`regularization_reference` / `regularization_weights` overrides can only be " *
                "applied to L2/L1/ElasticNetRegularizer (got a custom regularizer). Build the " *
                "overrides into the custom regularizer directly."
        )
    )
end

"""
    NonlinearSolve.NonlinearLeastSquaresProblem(problem::ParameterEstimationProblem)

Build the `NonlinearLeastSquaresProblem` (with optional `lb`/`ub`) driving the least-squares
fit of the layer's tunable parameters.

Trivial bounds (all entries `-Inf`/`Inf`) are omitted so that the resulting problem can be
solved with an unbounded solver such as `GaussNewton`/`LevenbergMarquardt`. This matters for
reverse-mode AD: passing any `lb`/`ub` (even trivial ones) makes NonlinearSolve wrap the
residual in a mutating bounds transformation that `AutoZygote` cannot differentiate. When
finite bounds are present, solve the returned problem with a `Bounded*` solver (e.g.
`BoundedGaussNewton`) so that reverse-mode AD keeps working.
"""
function SciMLBase.NonlinearLeastSquaresProblem(problem::ParameterEstimationProblem)
    # Use the stored out-of-place residual so NonlinearSolve can accurately infer the
    # residual dimension (number of observations + shooting constraints), which generally
    # differs from the number of fitted parameters. Being fully functional, it also allows
    # reverse-mode AD backends (e.g. `AutoZygote`) to differentiate the residual.
    (; residual_oop, x0, lb, ub) = problem
    fn = NonlinearFunction{false}(residual_oop)
    if _has_finite_bounds(lb) || _has_finite_bounds(ub)
        return NonlinearLeastSquaresProblem(fn, x0; lb = lb, ub = ub)
    end
    return NonlinearLeastSquaresProblem(fn, x0)
end

# Whether any bound is finite; `nothing` (no bounds) is treated as trivial.
_has_finite_bounds(::Nothing) = false
_has_finite_bounds(bounds) = any(isfinite, bounds)

"""
    NonlinearSolve.NonlinearLeastSquaresProblem(pe::ParameterEstimationLayer; kwargs...)

Construct the `NonlinearLeastSquaresProblem` for a [`ParameterEstimationLayer`](@ref) so it
can be solved with a NonlinearSolve least-squares solver, e.g.
`solve(prob, GaussNewton())` or `solve(prob, LevenbergMarquardt())`. Additional keywords are
forwarded to [`ParameterEstimationProblem`](@ref).
"""
function SciMLBase.NonlinearLeastSquaresProblem(
        pe::ParameterEstimationLayer; kwargs...
    )
    return NonlinearLeastSquaresProblem(ParameterEstimationProblem(pe; kwargs...))
end

"""
    problem(pe::ParameterEstimationLayer; kwargs...)

Return the `NonlinearSolve.NonlinearLeastSquaresProblem` for `pe` (an alias for
`NonlinearSolve.NonlinearLeastSquaresProblem(pe; kwargs...)`).
"""
problem(pe::ParameterEstimationLayer; kwargs...) =
    NonlinearLeastSquaresProblem(pe; kwargs...)
