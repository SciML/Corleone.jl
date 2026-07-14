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
    @info perm
    ps = sortperm(perm)
    return Controls(sys, nt, ps)
end

function (c::Controls)(t::T, ps, st::NamedTuple) where T <: Real 
    res, new_st = evaluate_controls(c.controls, t, ps.controls, st.controls,)
    return res[c.permutation], (; controls = new_st)
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

function collect_timegrid(controls, ps, st, tspan = (-Inf, Inf))
    tpoints = reduce(vcat, map(Base.Fix2(getfield, :tpoints), st.controls)) 
    filter!(ti->tspan[1] <= ti <= tspan[2], unique!(sort!(tpoints)))
    collect(zip(@view(tpoints[begin:end-1]), @view(tpoints[begin+1:end])))
    #ntuple(i->(tpoints[i], tpoints[i+1]), size(tpoints,1)-1)
end

@non_differentiable collect_timegrid(controls, ps, st, tspan)

function optimal_shooting_points(method::AbstractAutoShoot, layer::Controls, ps, st; timepoints = [])
    (; controls) = layer 
    tpoints = reduce(vcat, map(Base.Fix2(getfield, :tpoints), controls))
    append!(tpoints, timepoints)
    sort!(tpoints)
    unique!(tpoints)
    pattern = collect_activity_pattern(tpoints, layer, ps, st)
    next_shooting_points = apply_auto_shoot(method, pattern, tpoints)
    for layer in controls, ti in next_shooting_points
        inject!(layer, ti)
    end
    sort!(next_shooting_points)
    next_shooting_points
end
