module CorleoneChainRulesExtension

using Corleone
using ChainRulesCore
using Corleone: __remake_wrap, _resolve_index
using SymbolicIndexingInterface

ChainRulesCore.@non_differentiable SymbolicIndexingInterface.symbolic_container(::Any)
ChainRulesCore.@non_differentiable Corleone._resolve_index(::Any, ::Any)

"""
Custom rrule for `__remake_wrap` on `AbstractVector` buffers.

Forward: identical to the primal (non-mutating scatter).
Backward:
  • ∂oldbuffer : Δy with replaced positions zeroed out.
  • ∂vals      : Δy extracted at each replaced position.
"""
function ChainRulesCore.rrule(
    ::typeof(Corleone.__remake_wrap),
    sys,
    oldbuffer::AbstractVector,
    idxs,
    vals,
)
    int_idxs = map(idx -> _resolve_index(sys, idx), idxs)
    y = __remake_wrap(sys, oldbuffer, idxs, vals)

    proj_buf  = ProjectTo(oldbuffer)
    proj_vals = ProjectTo(vals)

    function remake_pullback(Δy)
        n = length(oldbuffer)
        # Gradient w.r.t. oldbuffer: pass through everywhere except replaced indices
        keep_flags = [i ∉ int_idxs for i in 1:n]
        Δoldbuffer = Δy .* keep_flags

        # Gradient w.r.t. vals: pick out the tangent at each replaced position
        Δvals = [Δy[pos] for pos in int_idxs]

        return (
            NoTangent(),   # __remake_wrap itself
            NoTangent(),   # sys
            proj_buf(Δoldbuffer),
            NoTangent(),   # idxs
            proj_vals(Δvals),
        )
    end

    return y, remake_pullback
end

end