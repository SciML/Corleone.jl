struct ObservedExpressionLayer{E,L,S,T,G} <: LuxCore.AbstractLuxLayer
    "Define if the layer represents an equality or inequality."
    kind::Symbol
    "The original expression provided by the user."
    expression::E
    "The lowered expression where time calls are replaced by index access."
    lowered::L
    "Signal names used by the expression in argument order for `compiled`."
    signal_names::S
    "Requested timepoints per signal."
    timepoints::T
    "Compiled callable function that evaluates the lowered expression."
    compiled::G
end

const _OP_EQ = :(==)
const _OP_LE = :(<=)
const _OP_GE = :(>=)

function _comparison_kind(op)
    if op === _OP_EQ
        return :eq
    elseif op === _OP_LE || op === Symbol("≤")
        return :le
    elseif op === _OP_GE || op === Symbol("≥")
        return :ge
    end
    error("Observed expressions must be equalities or inequalities (`==`, `<=`, `>=`).")
end

function _lower_comparison(expr::Expr)
    if expr.head != :call || length(expr.args) != 3
        error("Observed expression must be a binary equality or inequality.")
    end
    op = expr.args[1]
    kind = _comparison_kind(op)
    lhs = expr.args[2]
    rhs = expr.args[3]
    return kind, :($lhs - $rhs)
end

_extract_timepoints(arg) = nothing
_extract_timepoints(arg::Number) = [Float64(arg)]
_extract_timepoints(arg::AbstractVector{<:Number}) = Float64.(collect(arg))

function _extract_timepoints(arg::Expr)
    if arg.head == :vect && all(ai -> ai isa Number, arg.args)
        return Float64.(arg.args)
    end
    return nothing
end

_is_symbol_time_call(ex) = ex isa Expr && ex.head == :call && length(ex.args) == 2 && ex.args[1] isa Symbol

function _register_signal_timepoint!(
    signal_names::Vector{Symbol},
    timepoints::Dict{Symbol,Vector{Float64}},
    lookup::Dict{Symbol,Dict{Float64,Int}},
    sym::Symbol,
    tp::Float64,
)
    haskey(timepoints, sym) || begin
        push!(signal_names, sym)
        timepoints[sym] = Float64[]
        lookup[sym] = Dict{Float64,Int}()
    end
    id = get(lookup[sym], tp, 0)
    if id == 0
        push!(timepoints[sym], tp)
        id = length(timepoints[sym])
        lookup[sym][tp] = id
    end
    return id
end

function _lower_signal_time_calls(expr)
    signal_names = Symbol[]
    timepoints = Dict{Symbol,Vector{Float64}}()
    lookup = Dict{Symbol,Dict{Float64,Int}}()

    function _lower(ex)
        if _is_symbol_time_call(ex)
            sym = ex.args[1]
            tps = _extract_timepoints(ex.args[2])
            if !isnothing(tps)
                ids = map(tp -> _register_signal_timepoint!(signal_names, timepoints, lookup, sym, tp), tps)
                idx_expr = length(ids) == 1 ? first(ids) : Expr(:vect, ids...)
                return Expr(:ref, sym, idx_expr)
            end
        end
        if ex isa Expr
            return Expr(ex.head, map(_lower, ex.args)...)
        end
        return ex
    end

    lowered = _lower(expr)
    names = Tuple(signal_names)
    tps = NamedTuple{names}(Tuple(getindex.(Ref(timepoints), signal_names)))
    return lowered, names, tps
end

function _compile_lowered_expression(signal_names::NTuple{N,Symbol}, lowered) where {N}
    args = Expr(:tuple, signal_names...)
    return Core.eval(@__MODULE__, :($args -> $lowered))
end

function ObservedExpressionLayer(expression::Expr)
    kind, reduced = _lower_comparison(expression)
    lowered, signal_names, timepoints = _lower_signal_time_calls(reduced)
    compiled = _compile_lowered_expression(signal_names, lowered)
    return ObservedExpressionLayer{
        typeof(expression),
        typeof(lowered),
        typeof(signal_names),
        typeof(timepoints),
        typeof(compiled),
    }(kind, expression, lowered, signal_names, timepoints, compiled)
end

ObservedExpressionLayer(layer::ObservedExpressionLayer) = layer

LuxCore.parameterlength(::ObservedExpressionLayer) = 0
LuxCore.statelength(::ObservedExpressionLayer) = 0

function LuxCore.initialstates(::Random.AbstractRNG, layer::ObservedExpressionLayer)
    return (; timepoints=layer.timepoints, signal_names=layer.signal_names)
end

function _collect_signals(trajsignals::NamedTuple, indices::NamedTuple, signal_names::NTuple{N,Symbol}) where {N}
    return ntuple(i -> begin
        name = signal_names[i]
        trajsignals[name][indices[name]]
    end, Val(N))
end

function (layer::ObservedExpressionLayer)(traj::Trajectory, ps, st)
    trajsignals = map(st.getters) do getter
        getter(traj)
    end
    signals = _collect_signals(trajsignals, st.indices, st.signal_names)
    value = layer.compiled(signals...)
    return value, st
end

struct ObservedLayer{N,L,O} <: LuxCore.AbstractLuxContainerLayer{(:layer, :observed)}
    "The name of the layer, used for display and logging purposes."
    name::N
    "The wrapped shooting layer used to produce trajectories."
    layer::L
    "The user-provided expression as a NamedTuple of ObservedExpressionLayers."
    observed::O
end

function _existing_saveat(problem)
    t0, tinf = problem.tspan
    saveat = get(problem.kwargs, :saveat, Float64[])
    if isa(saveat, Number)
        return collect(t0:saveat:tinf)
    elseif isa(saveat, AbstractVector)
        return Float64.(saveat)
    end
    return Float64[]
end

function _sort_unique!(xs)
    unique!(sort!(xs))
    return xs
end

function _collect_control_timegrid(layer)
    tgrid = Float64[]
    if hasproperty(layer, :controls)
        append!(tgrid, Float64.(Corleone.get_timegrid(layer.controls)))
    end
    if hasproperty(layer, :layer)
        append!(tgrid, _collect_control_timegrid(getproperty(layer, :layer)))
    end
    if hasproperty(layer, :layers)
        foreach(values(layer.layers)) do lay
            append!(tgrid, _collect_control_timegrid(lay))
        end
    end
    return _sort_unique!(tgrid)
end

function _collect_expression_timepoints(observed::NamedTuple)
    tps = Float64[]
    foreach(values(observed)) do obs
        foreach(values(obs.timepoints)) do v
            append!(tps, v)
        end
    end
    return _sort_unique!(tps)
end

function _remake_layer_with_observed_grid(layer, observed)
    hasmethod(Corleone.get_problem, Tuple{typeof(layer)}) || return layer
    problem = Corleone.get_problem(layer)
    t0, tinf = problem.tspan
    requested = vcat(
        _existing_saveat(problem),
        _collect_expression_timepoints(observed),
        _collect_control_timegrid(layer),
    )
    requested = filter(tp -> (tp >= t0) && (tp <= tinf), requested)
    _sort_unique!(requested)
    if isempty(requested)
        return layer
    end
    if hasproperty(layer, :initial_conditions)
        problem_new = remake(problem; saveat=requested)
        initial_conditions = remake(layer.initial_conditions; problem=problem_new)
        return remake(layer; initial_conditions)
    end
    return remake(layer; saveat=requested)
end

function ObservedLayer(layer, observed::NamedTuple; name=gensym(:observed))
    observed_layers = map(ObservedExpressionLayer, observed)
    remade_layer = _remake_layer_with_observed_grid(layer, observed_layers)
    return ObservedLayer{typeof(name),typeof(remade_layer),typeof(observed_layers)}(name, remade_layer, observed_layers)
end

function (layer::ObservedLayer)(u0, ps, st)
    traj, layer_st = layer.layer(u0, ps.layer, st.layer)
    observed, observed_st = apply_observed(layer.observed, traj, ps.observed, st.observed)
    return (; trajectory = traj, observed = observed), (; layer=layer_st, observed=observed_st)
end

function _flatten_timestop_bins(timestops::Tuple)
    tgrid = mapreduce(vcat, timestops; init=Float64[]) do bin
        mapreduce(vcat, bin; init=Float64[]) do (t0, t1)
            [Float64(t0), Float64(t1)]
        end
    end
    return _sort_unique!(tgrid)
end

function get_timegrids(layer_st)
    if hasproperty(layer_st, :timestops)
        return _flatten_timestop_bins(layer_st.timestops)
    elseif layer_st isa NamedTuple
        tgrid = Float64[]
        foreach(values(layer_st)) do sti
            append!(tgrid, get_timegrids(sti))
        end
        return _sort_unique!(tgrid)
    end
    error("Could not extract timegrids from layer state.")
end

function _extract_system(layer_st)
    if hasproperty(layer_st, :system)
        return layer_st.system
    elseif layer_st isa NamedTuple
        return _extract_system(first(values(layer_st)))
    end
    error("Could not extract symbolic system from layer state.")
end

function _build_signal_getters(system, signal_names::NTuple{N,Symbol}) where {N}
    vals = ntuple(i -> begin
        getsym(system, signal_names[i])
    end, Val(N))
    return NamedTuple{signal_names}(vals)
end

function LuxCore.initialstates(rng::Random.AbstractRNG, layer::ObservedLayer)
    layer_st = LuxCore.initialstates(rng, layer.layer)
    timegrid = get_timegrids(layer_st)
    system = _extract_system(layer_st)
    observed_st = map(layer.observed) do obs
        st_ = LuxCore.initialstates(rng, obs)
        indices = find_time_indices(st_.timepoints, system, timegrid)
        getters = _build_signal_getters(system, st_.signal_names)
        merge(st_, (; indices, getters))
    end
    return (; layer=layer_st, observed=observed_st)
end

function find_time_indices(requested::NamedTuple{fields}, system, timegrid::AbstractVector{<:Real}) where {fields}
    isempty(timegrid) && error("Cannot map observed timepoints because the integration grid is empty.")
    gridder = Base.Fix1(searchsortedlast, timegrid)

    function _max_grid_index_for_signal(field)
        if is_timeseries_parameter(system, field)
            return max(firstindex(timegrid), lastindex(timegrid) - 1)
        elseif is_variable(system, field)
            return lastindex(timegrid)
        end
        error("Requested observed signal $field is neither a state variable nor a time-series parameter.")
    end

    vals = map(fields) do field
        ti = getproperty(requested, field)
        upper = _max_grid_index_for_signal(field)
        map(ti) do tij
            clamp(gridder(tij), firstindex(timegrid), upper)
        end
    end
    return NamedTuple{fields}(Tuple(vals))
end

@generated function apply_observed(observed::NamedTuple{fields}, traj::Trajectory, ps, st::NamedTuple{fields}) where {fields}
    sts = [gensym() for _ in fields]
    rets = [gensym() for _ in fields]
    exprs = Expr[]
    for (i, f) in enumerate(fields)
        push!(exprs, :(($(rets[i]), $(sts[i])) = observed.$(f)(traj, ps.$(f), st.$(f))))
    end
    push!(exprs, :(return (NamedTuple{fields}(($(rets...),)), NamedTuple{fields}(($(sts...),)))))
    return Expr(:block, exprs...)
end

