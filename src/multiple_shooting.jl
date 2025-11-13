"""
$(TYPEDEF)
Defines a callable layer that consists of several [``SingleShootingLayer``](@ref) collected
in `layers` that are applied on disjunct time intervals given in `shooting_intervals`.
Numerical integration of the differential equations of the layers is separated as
initial conditions are degrees of freedom (except perhaps for the first layer).
Thus, parallelization is possible, for which a suitable `ensemble_alg` can be specified.
Additionally, `bounds_nodes` define the bounds on the multiple shooting nodes.

# Fields
$(FIELDS)
"""
struct MultipleShootingLayer{L,SI,E,B,D} <: LuxCore.AbstractLuxLayer
    "Collection of multiple SingleShootingLayer"
    layers::L
    "Collection of tspans for individual layers"
    shooting_intervals::SI
    "Ensemble method to solve EnsembleProblem"
    ensemble_alg::E
    "Bounds on the multiple shooting nodes"
    bounds_nodes::B
    "Duplicated controls due to misalignment of control and MS grids."
    duplicate_controls::D
end

get_problem(layer::MultipleShootingLayer) = get_problem(first(layer.layers))
get_controls(layer::MultipleShootingLayer) = get_controls(first(layer.layers))
get_tspan(layer::MultipleShootingLayer) = (first(first(layer.layers).problem.tspan), last(last(layer.layers).problem.tspan))
get_tunable(layer::MultipleShootingLayer) = get_tunable(first(layer.layers))
get_params(layer::MultipleShootingLayer) = get_params(first(layer.layers))
get_bounds(layer::MultipleShootingLayer) = begin
    layer_names = Tuple([Symbol("layer_$i") for i=1:length(layer.layers)])
    layer_bounds = map(layer.layers) do _layer
        get_bounds(_layer)
    end
    ComponentArray(NamedTuple{layer_names}(first.(layer_bounds))), ComponentArray(NamedTuple{layer_names}(last.(layer_bounds)))
end
"""
    get_shooting_constraints(layer)
Returns function to evaluate matching conditions of MultpipleShootingLayer.
The signature of the function is (sols,p).

# Examples
```julia-repl
julia> shooting_constraint = get_shooting_constraints(layer)
julia> sols, _ = layer(nothing, p, st)
julia> shooting_constraint(sols, p)
[0.7626323782771849, 1.118278846482416, 1.2309540387687008, -3.0538497395734483, -2.6549931647655867, -0.4674885764266593]
```
"""
get_shooting_constraints(layer::MultipleShootingLayer) = begin
    ps, st = LuxCore.setup(Random.default_rng(), layer)
    ax = getaxes(ComponentArray(ps))
    controls, control_indices = get_controls(layer)
    matching = let ax = ax, nc = length(controls)
        (sols, p) -> begin
            _p = isa(p, Array) ? ComponentArray(p, ax) : p
            shooting = reduce(vcat, map(zip(sols[1:end-1], keys(ax[1])[2:end])) do (sol, name_i)
                _u0 = getproperty(_p, name_i).u0
                sol.u[end][1:end-nc] .-_u0
            end)
            controls = reduce(vcat, map(enumerate(layer.duplicate_controls)) do (i, duplicate)
                [p["layer_$(d.pre.i)"].controls[d.pre.idx] - p["layer_$(d.post.i)"].controls[d.post.idx] for d in duplicate]
            end)
            return vcat(shooting, controls)
        end
    end
    return matching
end


function compute_number_discretized_controls_so_far(controls, shooting_intervals, control_idx, interval; compensate_repeated_controls=true)
    sum(map(enumerate(controls)) do (_control_idx, _c)
        n_controls = sum(shooting_intervals[interval][1] .<= _c.t .< shooting_intervals[interval][2])
        if first(shooting_intervals[interval]) ∉ _c.t && compensate_repeated_controls
            n_controls += 1
        end
        if _control_idx > control_idx
            0
        else
            n_controls == 0 ? 1 : n_controls
        end
    end)
end


function compute_duplicate_controls(controls, shooting_intervals, continuous_controls=eachindex(controls))
    duplicates = map(enumerate(controls)) do (control_idx, c)
        if control_idx ∉ continuous_controls
            nothing
        else
            filter(!isnothing, map(enumerate(shooting_intervals)) do (i,tspani)
                lo, hi = tspani
                idx = findall(lo .<= c.t .< hi)
                if isempty(idx) # Case: No discretized controls in shooting interval
                    idx_pre = findlast(c.t .< lo)
                    interval = findfirst(map(t -> first(t) .<= c.t[idx_pre] .< last(t), shooting_intervals))
                    idx_on_layer_pre = compute_number_discretized_controls_so_far(controls, shooting_intervals, control_idx, interval)
                    idx_on_layer_post = compute_number_discretized_controls_so_far(controls, shooting_intervals, control_idx, i; compensate_repeated_controls=false)
                    (pre=(i=interval, idx=idx_on_layer_pre), post=(i=i,idx=idx_on_layer_post))
                elseif lo ∉ c.t
                    idx_pre = findlast(c.t .< lo)
                    interval = i - 1
                    idx_on_layer_pre = compute_number_discretized_controls_so_far(controls, shooting_intervals, control_idx, interval)
                    idx_on_layer_post = compute_number_discretized_controls_so_far(controls, shooting_intervals, control_idx, interval+1; compensate_repeated_controls=false)

                    (pre=(i=interval, idx=idx_on_layer_pre), post=(i=i,idx=idx_on_layer_post))
                else
                    nothing
                end
            end)
        end
    end
    return duplicates
end

"""
$(SIGNATURES)

Constructs a MultipleShootingLayer from an `AbstractDEProblem`. Argument `shooting_points`
denote start of shooting intervals, and bounds of shooting nodes can be specified via
`bounds_nodes`. Integration can be parallelized via providing a suitable `ensemble_alg`,
however, `EnsembleSerial()` is used per default.
See also [``SingleShootingLayer``](@ref) for information on further arguments.
"""
function MultipleShootingLayer(prob, alg, control_indices, controls, shooting_points;
                tunable_ic = Int64[], bounds_ic = (-Inf*ones(length(tunable_ic)), Inf*length(tunable_ic)),
                bounds_nodes = (-Inf * ones(length(prob.u0)), Inf*ones(length(prob.u0))),
                ensemble_alg = EnsembleSerial(), continuous_controls=eachindex(controls), kwargs...)
    tspan = prob.tspan
    shooting_points = vcat(tspan..., shooting_points) |> unique! |> sort!
    shooting_intervals = [(t0,t1) for (t0,t1) in zip(shooting_points[1:end-1], shooting_points[2:end])]
    _tunable = vcat([tunable_ic], [collect(1:length(prob.u0)) for _ in 1:length(shooting_intervals)])
    layers = [SingleShootingLayer(remake(prob, tspan = tspani, kwargs...), alg, control_indices, restrict_controls(controls, tspani..., continuous_controls);
                tunable_ic=_tunable[i], bounds_ic = (i == 1 ? (isempty(tunable_ic) ? nothing : bounds_ic) : bounds_nodes)) for (i, tspani) in enumerate(shooting_intervals)]

    duplicates = compute_duplicate_controls(controls, shooting_intervals, continuous_controls)

    MultipleShootingLayer{typeof(layers), typeof(shooting_intervals), typeof(ensemble_alg), typeof(bounds_nodes), typeof(duplicates)}(layers, shooting_intervals, ensemble_alg, bounds_nodes, duplicates)
end


function LuxCore.initialparameters(rng::Random.AbstractRNG, mslayer::MultipleShootingLayer)
    layer_names = Tuple([Symbol("layer_$i") for i=1:length(mslayer.layers)])
    layer_ps    = Tuple([LuxCore.initialparameters(rng, layer) for layer in mslayer.layers])
    NamedTuple{layer_names}(layer_ps)
end

function LuxCore.initialstates(rng::Random.AbstractRNG, mslayer::MultipleShootingLayer)
    layer_names = Tuple([Symbol("layer_$i") for i=1:length(mslayer.layers)])
    layer_st    = Tuple([LuxCore.initialstates(rng, layer) for layer in mslayer.layers])
    NamedTuple{layer_names}(layer_st)
end

function (layer::MultipleShootingLayer)(::Any, ps, st)
    prob = SingleShootingProblem(first(layer.layers), ps.layer_1, st.layer_1)
    remaker = let ps = ps, st=st, names = keys(ps)
        function (prob, i, repeat)
            current = names[i]
            p_current = getproperty(ps, current)
            st_current = getproperty(st, current)
            prob_current = remake(prob; ps=p_current, st=st_current)
            prob_current
        end
    end
    return solve(EnsembleProblem(prob, prob_func=remaker, output_func = (sol, i) -> (sol[1], false)),
            DummySolve(),layer.ensemble_alg; trajectories = length(layer.layers)), st
end

"""
    get_block_structure(layer)

Compute the block structure of the hessian of the Lagrangian of an optimal control problem
as specified via the `shooting_intervals` of the `MultipleShootingLayer`.
"""
function get_block_structure(layer::MultipleShootingLayer)
    ps_lengths = map(LuxCore.parameterlength, layer.layers)
    vcat(0, cumsum(ps_lengths))
end

"""
    merge_ms_controls(layer)

Merges corresponding control definitions of the several `SingleShootingLayer` layers
collected in the `MultipleShootingLayer` into one control definition.
"""
function merge_ms_controls(layer::MultipleShootingLayer)
    nc = length(layer.layers[1].controls)

    map(1:nc) do i
        defs_control = map(layer.layers) do _l
            ci = _l.controls[i]

            (get_timegrid(ci), get_controls(Random.default_rng(), ci), get_bounds(ci))
        end
        name = first(layer.layers).controls[i].name
        new_timegrid = reduce(vcat, first.(defs_control))
        new_controls = reduce(vcat, [x[2] for x in defs_control])
        new_bounds = (reduce(vcat, first.(last.(defs_control))), reduce(vcat, last.(last.(defs_control))))
        ControlParameter(new_timegrid, name=name, controls=new_controls, bounds=new_bounds)
    end
end