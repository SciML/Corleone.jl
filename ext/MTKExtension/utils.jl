maybeop(x) = iscall(x) ? operation(x) : x

function Corleone.retrieve_symbol_cache(cache::ModelingToolkit.System, u0, p, control_indices; kwargs...) 
    x = unknowns(cache)
    p = tunable_parameters(cache)
    iv = ModelingToolkit.get_iv(cache)
    u = filter(ModelingToolkit.isinput, p)
    sort!(u, by = ui -> SymbolicIndexingInterface.parameter_index(cache, ui).index)
    p = filter(!ModelingToolkit.isinput, p)
    x = [x..., u...]
    @info p 
    SymbolCache(x, p, [iv])
end

function collect_timepoints!(tpoints, ex)
    if iscall(ex)
        op, args = operation(ex), arguments(ex)
        if SymbolicUtils.issym(op) && isa(first(args).val, Number) && length(args) == 1
            tp = first(args).val
            vars = get!(tpoints, op, typeof(tp)[]) 
            push!(vars, tp)
        end
        return op(map(args) do x 
            collect_timepoints!(tpoints, x)
        end...)
    end 
    return ex 
end

function collect_integrals!(subs, ex, t)
    if iscall(ex)
        op, args = operation(ex), arguments(ex)
        ex = op(
            map(args) do arg
                collect_integrals!(subs, arg, t)
            end...
        )
        if isa(op, Symbolics.Integral)
            var = get!(subs, ex) do 
                sym = Symbol(:𝕃, Symbol(Char(0x2080 + length(subs) + 1)))
                var = Symbolics.unwrap(only(ModelingToolkit.@variables ($sym)(t) = 0.0 [tunable = false, bounds = (0., Inf)])) # [costvariable = true]))
                var 
            end
            lo, hi = op.domain.domain.left, op.domain.domain.right
            return operation(var)(hi) - operation(var)(lo)
        end
    end
    return ex
end

Base.getindex(T::Corleone.Trajectory, ind::Num) = getindex(T, Symbolics.unwrap(ind))
function Base.getindex(T::Corleone.Trajectory, ind::SymbolicUtils.BasicSymbolic)
    if ind in keys(T.sys.variables)
        return vcat(getindex.(T.u, T.sys.variables[ind]))
    elseif ind in keys(T.sys.parameters)
        return getindex(T.p, T.sys.parameters[ind])
    end
    error(string("Invalid index: :", ind))
end
