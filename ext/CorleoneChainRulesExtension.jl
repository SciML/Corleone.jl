module CorleoneChainRulesExtension

using Corleone
using ChainRulesCore

function ChainRulesCore.rrule(::typeof(Corleone.__remake_wrap), sys, p, idxs, vals)
    y = __remake_wrap(sys, p, idxs, vals)
    getter = getp(sys, idxs)
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