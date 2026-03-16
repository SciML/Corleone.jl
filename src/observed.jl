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

function _comparison_kind(op)
    if op === :(==)
        return :eq
    elseif op === :(<=) || op === Symbol("≤")
        return :le
    elseif op === :(>=) || op === Symbol("≥")
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

_numbers_from_arg(arg) = nothing
_numbers_from_arg(arg::Number) = [Float64(arg)]
_numbers_from_arg(arg::AbstractVector{<:Number}) = Float64.(collect(arg))

function _numbers_from_arg(arg::Expr)
    if arg.head == :vect && all(ai -> ai isa Number, arg.args)
        return Float64.(arg.args)
    end
    return nothing
end

function _register_timepoint!(
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

function _lower_time_calls(expr)
    signal_names = Symbol[]
    timepoints = Dict{Symbol,Vector{Float64}}()
    lookup = Dict{Symbol,Dict{Float64,Int}}()

    function _lower(ex)
        if ex isa Expr && ex.head == :call && length(ex.args) == 2 && ex.args[1] isa Symbol
            sym = ex.args[1]
            tps = _numbers_from_arg(ex.args[2])
            if !isnothing(tps)
                ids = map(tp -> _register_timepoint!(signal_names, timepoints, lookup, sym, tp), tps)
                if length(ids) == 1
                    return Expr(:ref, sym, first(ids))
                end
                return Expr(:ref, sym, Expr(:vect, ids...))
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

function _compile_observed(signal_names::NTuple{N,Symbol}, lowered) where {N}
    args = Expr(:tuple, signal_names...)
    return Core.eval(@__MODULE__, :($args -> $lowered))
end

function ObservedExpressionLayer(expression::Expr)
    kind, reduced = _lower_comparison(expression)
    lowered, signal_names, timepoints = _lower_time_calls(reduced)
    compiled = _compile_observed(signal_names, lowered)
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

function _getsignals(trajsignals::NamedTuple, indices::NamedTuple, signal_names::NTuple{N,Symbol}) where {N}
    return ntuple(i -> begin
        name = signal_names[i]
        trajsignals[name][indices[name]]
    end, Val(N))
end

function (layer::ObservedExpressionLayer)(traj::Trajectory, ps, st)
    trajsignals = map(st.getters) do getter
        getter(traj)
    end
    signals = _getsignals(trajsignals, st.indices, st.signal_names)
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

function _get_existing_saveat(problem)
    t0, tinf = problem.tspan
    saveat = get(problem.kwargs, :saveat, Float64[])
    if isa(saveat, Number)
        return collect(t0:saveat:tinf)
    elseif isa(saveat, AbstractVector)
        return Float64.(saveat)
    end
    return Float64[]
end

function _get_control_timegrid(layer)
    if hasproperty(layer, :controls)
        return Float64.(Corleone.get_timegrid(layer.controls))
    elseif hasproperty(layer, :layer)
        return _get_control_timegrid(getproperty(layer, :layer))
    elseif hasproperty(layer, :layers)
        tgrid = mapreduce(vcat, values(layer.layers); init=Float64[]) do lay
            _get_control_timegrid(lay)
        end
        unique!(sort!(tgrid))
        return tgrid
    end
    return Float64[]
end

function _all_expression_timepoints(observed::NamedTuple)
    tps = Float64[]
    foreach(values(observed)) do obs
        foreach(values(obs.timepoints)) do v
            append!(tps, v)
        end
    end
    unique!(sort!(tps))
    return tps
end

function _remake_observed_layer(layer, observed)
    hasmethod(Corleone.get_problem, Tuple{typeof(layer)}) || return layer
    problem = Corleone.get_problem(layer)
    t0, tinf = problem.tspan
    requested = vcat(
        _get_existing_saveat(problem),
        _all_expression_timepoints(observed),
        _get_control_timegrid(layer),
    )
    requested = filter(tp -> (tp >= t0) && (tp <= tinf), requested)
    unique!(sort!(requested))
    if isempty(requested)
        return layer
    end
    if hasproperty(layer, :initial_conditions)
        problem = Corleone.get_problem(layer)
        problem_new = remake(problem; saveat=requested)
        initial_conditions = remake(layer.initial_conditions; problem=problem_new)
        return remake(layer; initial_conditions)
    end
    return remake(layer; saveat=requested)
end

function ObservedLayer(layer, observed::NamedTuple; name=gensym(:observed))
    observed_layers = map(ObservedExpressionLayer, observed)
    remade_layer = _remake_observed_layer(layer, observed_layers)
    return ObservedLayer{typeof(name),typeof(remade_layer),typeof(observed_layers)}(name, remade_layer, observed_layers)
end

function (layer::ObservedLayer)(u0, ps, st)
    traj, layer_st = layer.layer(u0, ps.layer, st.layer)
    observed, observed_st = apply_observed(layer.observed, traj, ps.observed, st.observed)
    return observed, (; layer=layer_st, observed=observed_st)
end

function _flatten_timestops(timestops::Tuple)
    tgrid = mapreduce(vcat, timestops; init=Float64[]) do bin
        mapreduce(vcat, bin; init=Float64[]) do (t0, t1)
            [Float64(t0), Float64(t1)]
        end
    end
    unique!(sort!(tgrid))
    return tgrid
end

function get_timegrids(layer_st)
    if hasproperty(layer_st, :timestops)
        return _flatten_timestops(layer_st.timestops)
    elseif layer_st isa NamedTuple
        tgrid = mapreduce(vcat, values(layer_st); init=Float64[]) do sti
            get_timegrids(sti)
        end
        unique!(sort!(tgrid))
        return tgrid
    end
    error("Could not extract timegrids from layer state.")
end

function _get_system(layer_st)
    if hasproperty(layer_st, :system)
        return layer_st.system
    elseif layer_st isa NamedTuple
        return _get_system(first(values(layer_st)))
    end
    error("Could not extract symbolic system from layer state.")
end

function _build_getters(system, signal_names::NTuple{N,Symbol}) where {N}
    vals = ntuple(i -> begin
        getsym(system, signal_names[i])
    end, Val(N))
    return NamedTuple{signal_names}(vals)
end

function LuxCore.initialstates(rng::Random.AbstractRNG, layer::ObservedLayer)
    layer_st = LuxCore.initialstates(rng, layer.layer)
    timegrid = get_timegrids(layer_st)
    system = _get_system(layer_st)
    observed_st = map(layer.observed) do obs
        st_ = LuxCore.initialstates(rng, obs)
        indices = find_time_indices(st_.timepoints, system, timegrid)
        getters = _build_getters(system, st_.signal_names)
        merge(st_, (; indices, getters))
    end
    return (; layer=layer_st, observed=observed_st)
end

function find_time_indices(requested::NamedTuple{fields}, system, timegrid::AbstractVector{<:Real}) where {fields}
    isempty(timegrid) && error("Cannot map observed timepoints because the integration grid is empty.")
    gridder = Base.Fix1(searchsortedlast, timegrid)
    vals = map(fields) do field
        ti = getproperty(requested, field)
        if is_timeseries_parameter(system, field)
            upper = max(firstindex(timegrid), lastindex(timegrid) - 1)
            map(ti) do tij
                clamp(gridder(tij), firstindex(timegrid), upper)
            end
        elseif is_variable(system, field)
            map(ti) do tij
                clamp(gridder(tij), firstindex(timegrid), lastindex(timegrid))
            end
        else
            error("Requested observed signal $field is neither a state variable nor a time-series parameter.")
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

