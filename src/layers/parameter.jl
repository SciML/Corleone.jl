"""
$(TYPEDEF)
"""
@concrete terse struct PiecewiseConstant <: LuxCore.AbstractLuxContainerLayer{(:lookup, :parameter)}
    ""
    name
    ""
    lookup
    ""
    parameter
end

function PiecewiseConstant(timegrid::AbstractVector{<:Number}, default; name = nothing, cache = nothing, kwargs...)
    lookup = LookUp(timegrid, cache)
    parameters = ParameterCollection((rng, outdims) -> fill(default, outdims), length(timegrid) + 1)
    return PiecewiseConstant(name, lookup, parameters)
end

function PiecewiseConstant(timegrid::AbstractVector{<:Number}, default::Base.Callable; name = nothing, cache = nothing, kwargs...)
    lookup = LookUp(timegrid, cache)
    parameters = ParameterCollection(default, length(timegrid) + 1)
    return PiecewiseConstant(name, lookup, parameters)
end

function PiecewiseConstant(timegrid::Nothing, default::Base.Callable; name = nothing, cache = nothing, kwargs...)
    lookup = LookUp(timegrid, cache)
    parameters = ParameterCollection(default, 1)
    return PiecewiseConstant(name, lookup, parameters)
end

function PiecewiseConstant(timegrid::Nothing, default; name = nothing, cache = nothing, kwargs...)
    lookup = LookUp(timegrid, cache)
    parameters = ParameterCollection((rng, outdims) -> fill(default, outdims), 1)
    return PiecewiseConstant(name, lookup, parameters)
end

function (pc::PiecewiseConstant)(t, ps, st)
    (; lookup, parameter) = pc
    (idx, inserted), lookup_st = LuxCore.apply(lookup, t, ps.lookup, st.lookup)
    params, parameter_st = LuxCore.apply(parameter, idx, ps.parameter, st.parameter)
    return params, (; lookup = lookup_st, parameter = parameter_st)
end

get_timestops(pc::PiecewiseConstant, ps, st, ::Type{T}) where {T} = get_timestops(pc.lookup, ps.lookup, st.lookup, T)

@concrete terse struct Parameter <: LuxCore.AbstractLuxContainerLayer{(:parameters,)}
    ""
    name
    ""
    repack
    ""
    parameters
end

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

process_definition(::Any, ::Val{X}, k::Int, v) where {X} = k => v

process_definition(y, x::Val{X}, ks::Tuple, v) where {X} = reduce(
    vcat, map(ks) do k
        process_definition(y, x, k, v)
    end
)

process_definition(y, x::Val{X}, ks::AbstractArray, v) where {X} = map(ks) do k
    process_definition(y, x, k, v)
end

function process_definition(indep, ::Val{X}, k::Any, v) where {X}
    idx = if X === :parameter
        SymbolicIndexingInterface.parameter_index(indep, k)
    elseif X === :variable
        SymbolicIndexingInterface.variable_index(indep, k)
    end
    return idx => v
end

function Parameter(
        prob::SciMLBase.AbstractSciMLProblem,
        defintions...;
        name = nothing,
        fixed = [],
    )
    sys = get_symbolic_container(prob)
    # Int => Timepoints
    defintions = reduce(
        vcat, map(defintions) do def
            process_definition(sys, Val(:parameter), def...)
        end
    )
    # Fill out missings
    missing_defs = setdiff(
        [SymbolicIndexingInterface.parameter_index(sys, k) for k in SymbolicIndexingInterface.parameter_symbols(sys)],
        first.(defintions)
    ) .=> nothing
    defintions = vcat(defintions, missing_defs)
    sort!(defintions, by = first)
    vals, repack, _ = SciMLStructures.canonicalize(SciMLStructures.Tunable(), prob.p)
    psyms = SymbolicIndexingInterface.parameter_symbols(sys)
    psymmap = Dict([SymbolicIndexingInterface.parameter_index(sys, k) => k for k in psyms])
    fixed = map(fixed) do k
        SymbolicIndexingInterface.parameter_index(sys, k)
    end
    parameters = map(defintions) do (k, v)
        pname = psymmap[k]
        layer = PiecewiseConstant(v, vals[k]; name = pname)
        Symbol(pname) => k ∈ fixed ? FixedLayer(layer) : layer
    end |> NamedTuple
    return Parameter(name, repack, parameters)
end


function get_timestops(pc::Parameter, ps, st, ::Type{T} = Float64) where {T}
    idxs = findall(x -> typeof(x) <: PiecewiseConstant, pc.parameters)
    tstops = reduce(
        vcat, map(idxs) do id
            get_timestops(pc.parameters[id], ps.parameters[id], st.parameters[id], T)
        end
    )
    sort!(tstops)
    unique!(tstops)
    return tstops
end

function (layer::Parameter)(t, ps, st)
    (; repack, parameters) = layer
    out, new_st = apply_parameters(parameters, t, ps.parameters, st.parameters)
    return repack(out), (; parameters = new_st)
end

@generated function apply_parameters(params::NamedTuple{names}, t, ps, st::NamedTuple{names}) where {names}
    outs = Symbol[]
    sts = Symbol[]
    exprs = Expr[]
    for n in names
        push!(outs, gensym(:out))
        push!(sts, gensym(:st))
        push!(
            exprs,
            :(($(last(outs)), $(last(sts))) = params.$(n)(t, ps.$(n), st.$(n)))
        )
    end
    push!(
        exprs,
        :(new_st = NamedTuple{$(names)}(($(sts...),)))
    )
    push!(
        exprs,
        :(vals = vcat($(outs...)))
    )
    push!(exprs, :(return vals, new_st))
    return Expr(:block, exprs...)
end

(layer::Parameter)(prob::SciMLBase.AbstractDEProblem, ps, st) = layer(first(prob.tspan), ps, st)
(layer::Parameter)(sol::SciMLBase.AbstractTimeseriesSolution, ps, st) = layer(last(current_time(sol)), ps, st)
