module CorleoneChainRulesExtension

using Corleone
using ChainRulesCore
using Corleone: __remake_wrap
using SymbolicIndexingInterface

ChainRulesCore.@non_differentiable SymbolicIndexingInterface.symbolic_container(x)

function ChainRulesCore.rrule(::typeof(Corleone.__remake_wrap), sys, p, idxs, vals)
    y = __remake_wrap(sys, copy(p), idxs, vals)
    getter = getsym(sys, idxs)
    proj_p = ProjectTo(p)
    proj_vals = ProjectTo(vals)
    function remake_pullback(Δy)
        Δvals = getter(Δy)
        Δp = remake_buffer(sys, copy(Δy), idxs, zero.(vals))
        return NoTangent(), NoTangent(), proj_p(Δp), NoTangent(), proj_vals(Δvals)
    end
    return y, remake_pullback
end
end