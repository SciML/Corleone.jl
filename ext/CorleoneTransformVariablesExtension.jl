module CorleoneTransformVariablesExtension
import Corleone
using TransformVariables: TransformVariables, inverse, transform, transform_and_logjac

const ScalarTransform = TransformVariables.ScalarTransform

Corleone.natural_value(t::ScalarTransform, x) = transform(t, x)
Corleone.internal_value(t::ScalarTransform, x) = inverse(t, x)
Corleone.log_jacobian(t::ScalarTransform, x) = last(transform_and_logjac(t, x))
# These maps enforce their own natural domain; optimizer coordinates are free.
Corleone.get_bounds(::ScalarTransform) = (-Inf, Inf)
# The open image of the real line; its endpoints have no finite internal value.
Corleone.in_domain(t::ScalarTransform, x) = transform(t, -Inf) < x < transform(t, Inf)

function Corleone.validate_transform(t::T where {T <: ScalarTransform})
    transform(t, 1.0) > transform(t, 0.0) || throw(ArgumentError("Parameter transforms must be increasing."))
    return nothing
end

end
