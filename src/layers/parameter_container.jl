@concrete terse struct Controls{N <: NamedTuple} <: LuxCore.AbstractLuxContainerLayer{(:controls,)}
    sys
    controls::N 
    permutation
end

maybekeys(x, name) = SymbolicIndexingInterface.hasname(x) ? SymbolicIndexingInterface.getname(x) : gensym(name)

function Controls(x...; sys = nothing, kwargs...)
    NAMES = LuxCore.display_name.(x)
    nt = NamedTuple{NAMES}(x)
    perm = reduce(vcat, map(x) do xi 
        get_parameter_index(sys, xi)
    end)
    ps = sortperm(perm)
    return Controls(sys, nt, ps)
end

function (c::Controls)(t::T, ps, st::NamedTuple) where T <: Real 
    res, new_st = evaluate_controls(c.controls, t, ps.controls, st.controls,)
    return res[c.permutation], new_st
end

@generated function evaluate_controls(controls::NamedTuple{NAMES}, t, ps, st::NamedTuple{NAMES}) where NAMES
    returns = [gensym(:res) for i in NAMES]
    sts = [gensym(:st) for i in NAMES]
    expr = Expr[]
    for (i,n) in enumerate(NAMES) 
        push!(expr, 
            :(($(returns[i]), $(sts[i])) = controls.$(n)(t, ps.$(n), st.$(n)))
        )
    end
    push!(expr, :(res = $(Expr(:call, reduce, vcat, Expr(:vect, returns...)))))
    push!(expr, :(st = NamedTuple{$(NAMES)}(($(sts...),))))
    push!(expr, :(return (res, st)))
    return Expr(:block, expr...)
end
