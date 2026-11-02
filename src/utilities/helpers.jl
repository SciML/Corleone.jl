function __get_symbolic_container(x)
    hasmethod(SymbolicIndexingInterface.symbolic_container, Tuple{typeof(x)}) || return x
    y = SymbolicIndexingInterface.symbolic_container(x)
    y != x && return __get_symbolic_container(y)
    return x
end

get_symbolic_container(x) = x

function get_symbolic_container(x::SciMLBase.AbstractSciMLProblem)
    indp = __get_symbolic_container(x)
    isa(indp, SymbolCache{Nothing, Nothing, Nothing, Nothing}) || return indp
    return make_default_sys(x)
end

_subscript(i::Integer) = (i |> digits |> reverse .|> dgt -> Char(0x2080 + dgt)) |> join

function make_default_sys(x::SciMLBase.AbstractDEProblem)
    default_vars = map(eachindex(IndexLinear(), x.u0)) do i
        Symbol(:u, _subscript(i)) => x.u0[i]
    end
    default_params = map(eachindex(IndexLinear(), x.p)) do i
        Symbol(:p, _subscript(i)) => x.p[i]
    end

    defaults = Dict(vcat(default_vars, default_params))

    return SymbolCache(
        first.(default_vars),
        first.(default_params),
        :t;
        defaults
    )
end
