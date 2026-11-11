"""
    BoundedTransform(lowerbound = -Inf, upperbound = Inf)

Identity mapping with explicit optimization bounds. Forward and inverse mappings
preserve values, and the log-Jacobian is zero. In unresolved annotations, `nothing`
bounds inherit symbolic metadata. Explicit transforms replace this default object
and supply their own optimization bounds.
"""
@concrete terse struct BoundedTransform
    lowerbound
    upperbound
end
BoundedTransform() = BoundedTransform(-Inf, Inf)
BoundedTransform(lowerbound) = BoundedTransform(lowerbound, Inf)

natural_value(::BoundedTransform, x) = x
internal_value(::BoundedTransform, x) = x
log_jacobian(::BoundedTransform, x) = zero(x)
"""
    get_bounds(transform::BoundedTransform)

Return the bounds stored by the identity transform. TransformVariables transforms
implement this interface through the extension, returning `(-Inf, Inf)`.
"""
get_bounds(t::BoundedTransform) = (t.lowerbound, t.upperbound)
validate_transform(::BoundedTransform) = nothing
validate_transform(t) = throw(ArgumentError("Unsupported parameter transform $(typeof(t)); load TransformVariables for scalar transforms."))

# Preserve identity buffers by reference, without mapping or allocating.
natural_values(::BoundedTransform, x) = x
natural_values(t, x) = x isa Number ? natural_value(t, x) : map(xi -> natural_value(t, xi), x)
internal_values(::BoundedTransform, x) = x
internal_values(t, x) = map(xi -> internal_value(t, xi), x)

# Shooting layers retain live symbolic-bound overrides for the identity mapping.
# Other transforms own their bounds and do not inherit additional constraints.
internal_bounds(::BoundedTransform, bounds) = bounds
internal_bounds(t, bounds) = get_bounds(t)

log_density(::Nothing, x) = zero(x)

# Constructor normalization keeps transforms as independent implementations of the
# same protocol. Explicit transforms replace bound keywords; only the default
# identity stores them. Grouped annotations may contain a different map per entry.
annotation_transform(::Nothing, lower, upper) = BoundedTransform(lower, upper)
annotation_transform(t, lower, upper) = t
function annotation_transform(ts::AbstractVector, lower, upper)
    check_transform_bound_lengths(ts, lower, upper)
    return map(eachindex(ts)) do i
        annotation_transform(ts[i], bound_entry(lower, i), bound_entry(upper, i))
    end
end
bound_entry(x, i) = x isa AbstractVector ? x[i] : x
function check_transform_bound_lengths(ts, lower, upper)
    for bound in (lower, upper)
        bound isa AbstractVector && length(bound) != length(ts) &&
            throw(DimensionMismatch("Grouped bounds must match the transform count."))
    end
    return nothing
end

annotation_bounds(t) = get_bounds(t)
annotation_bounds(ts::AbstractVector) = (map(t -> first(get_bounds(t)), ts), map(t -> last(get_bounds(t)), ts))

replace_transform_bounds(t, lower, upper) = t
replace_transform_bounds(t::BoundedTransform, lower, upper) = BoundedTransform(
    ismissing(lower) ? t.lowerbound : lower, ismissing(upper) ? t.upperbound : upper
)
function replace_transform_bounds(ts::AbstractVector, lower, upper)
    check_transform_bound_lengths(ts, lower, upper)
    return map(eachindex(ts)) do i
        replace_transform_bounds(ts[i], bound_entry(lower, i), bound_entry(upper, i))
    end
end
resolve_transform(t, fallback) = t
resolve_transform(t::BoundedTransform, (lower, upper)) = BoundedTransform(
    isnothing(t.lowerbound) ? lower : t.lowerbound,
    isnothing(t.upperbound) ? upper : t.upperbound,
)

"""
$(SIGNATURES)

Return the log prior density of the tunable parameters `ps` of `layer`. Each value of
a collection with a prior contributes `logpdf(prior, θ)` at its natural value `θ`,
plus the log-Jacobian of its transform, so the density refers to the stored internal
values. Collections without a prior contribute nothing.
"""
function log_prior end
