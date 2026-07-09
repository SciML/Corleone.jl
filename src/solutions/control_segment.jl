"""
$(TYPEDEF)

A `ControlSegment` represents one piecewise-constant control interval. It stores the
state trajectory `u` at the saved timepoints, the full parameter vector `p` (which
includes the constant control value), the timepoints `t`, and a `ControlSymbolCache`
`sys` that enables symbolic indexing.
"""
@concrete terse struct ControlSegment
    "The underlying DESolution"
    sol
    "Symbolic system cache"
    sys
end

get_utype(seg::ControlSegment) = Base.promote_eltype(seg.sol.u[1], control_values(seg))

SymbolicIndexingInterface.symbolic_container(seg::ControlSegment) = seg.sys

function control_values(seg::ControlSegment)
    cache = symbolic_container(seg)
    syms = sort(collect(keys(cache.controls)); by=s -> cache.controls[s][1])
    return [parameter_values(seg.sol)[cache.controls[s][2]] for s in syms]
end


function SymbolicIndexingInterface.state_values(seg::ControlSegment)
    c = control_values(seg)
    return vcat.(seg.sol.u, fill(c, length(seg.sol.t)))
end


function minimal_state_values(seg::ControlSegment)
    q_idxs = quadrature_indices(seg.sys)
    n = length(first(seg.sol.u))
    keep = setdiff(1:n, q_idxs)
    return [u_i[keep] for u_i in seg.sol.u]
end

SymbolicIndexingInterface.parameter_values(seg::ControlSegment) = begin 
    (; sys, sol) = seg
    map(Base.Fix1(parameter_values, sol), sort((collect ∘ values)(sys.parameter_indices)))
end

SymbolicIndexingInterface.current_time(seg::ControlSegment) = seg.sol.t
