module CorleoneComponentArraysExtension
using Corleone
using ComponentArrays

struct CAFunctionWrapper{A, F} <: Corleone.AbstractCorleoneFunctionWrapper
    "The axes of the componentarray"
    ax::A
    "The original function"
    f::F
end

(f::CAFunctionWrapper)(ps, st) = f.f(ComponentArray(ps, f.ax), st)
(f::CAFunctionWrapper)(res, ps, st) = f.f(res, ComponentArray(ps, f.ax), st)

Corleone.to_vec(::CAFunctionWrapper, x...) = map(x -> isnothing(x) ? x : (collect ∘ ComponentArray)(x), x)

function Corleone.wrap_functions(::Val{:ComponentArrays}, u0::NamedTuple, f...)
    u0 = ComponentVector(u0)
    ax = getaxes(u0)
    return map(f) do fi
        isnothing(fi) ? fi : CAFunctionWrapper{typeof(ax), typeof(fi)}(ax, fi)
    end
end

end
