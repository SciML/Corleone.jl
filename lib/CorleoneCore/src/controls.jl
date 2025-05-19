# Piecewise constant control function 
function find_index(t, val)
    idx = searchsortedlast(t, val)
    # We always assume that we have left / right continuity
    min(max(firstindex(t), idx), lastindex(t))
end

function find_index(p, val, t)
    @assert size(t, 1) == size(p, 1) "The dimensionality of the provided `tstops` and `parameters` are not consistent."
    id = find_index(t, val)
    getindex(p, id)
end

"""
$(TYPEDEF)

A struct representing a piecewise constant control signal. 

# Fields 
$(FIELDS)
"""

struct Control{T} <: Function
    "The time stops, e.g. switching points, of the signal"
    tstops::T
end

(c::Control)(p::AbstractVector, t::T) where {T} = find_index(p, t, c.tstops)

@register_symbolic (c::Control)(p::AbstractVector, t::Real)::Real

# TODO This is also a valid approach 
#function piecewise_constant(t, ts, ps)
#    @assert size(ts, 1) + 1 == size(ps, 1)
#    eq = last(ps)
#    for i in reverse(axes(ts,1))
#        eq = ifelse(t < ts[i],  ps[i], eq)
#    end
#    eq
#end

# Extend the current system to discrete controls 
function extend_discrete_controls(sys)
    t = independent_variable(sys)
    u = filter(is_discretecontrol, unknowns(sys))
    new_params = []
    eqs = Equation[]
    foreach(u) do ui
        usym = Symbol(Symbolics.operation(ui), :ₜ)
        ts = get_timepoints(ui)
        ps0 = [Symbolics.getdefaultval(ui) for _ in Base.OneTo(length(ts))]
        c = Control(ts)
        psym = Symbol(Symbolics.operation(ui), :ᵢ)
        lower, upper = getbounds(ui)
        pnew = @parameters begin
            ($(usym)::typeof(c))(..) = c
            ($psym)[1:length(ts)] = ps0, [bounds = (lower, upper)]
        end
        lhs = pnew[1](pnew[2], t)
        if is_differentialinput(ui)
            push!(eqs, D(ui) ~ lhs)
        else
            push!(eqs, ui ~ lhs)
        end
        append!(new_params, pnew)
    end
    control = ODESystem(Symbolics.scalarize.(eqs), t, [], new_params, name=Symbol(:control, :_, nameof(sys)), tspan = ModelingToolkit.get_tspan(sys))
    extend(sys, control)
end
