struct MultipleShootingLayer{L,I,E} <: LuxCore.AbstractLuxWrapperLayer{:layer}
  "The original layer"
  layer::L
  "The shooting intervals"
  shooting_intervals::I
  "The ensemble algorithm"
  ensemble_alg::E
end

function MultipleShootingLayer(layer, tpoints::Real...; ensemble_alg=EnsembleSerial(), kwargs...)
  tspans = vcat(collect(tpoints), collect(layer.problem.tspan))
  sort!(tspans)
  unique!(tspans)
  tspans = [tispan for tispan in zip(tspans[1:end-1], tspans[2:end])]
  tspans = tuple(tspans...)
  MultipleShootingLayer{typeof(layer),typeof(tspans),typeof(ensemble_alg)}(
    layer, tspans, ensemble_alg
  )
end

function LuxCore.initialparameters(rng::Random.AbstractRNG, shooting::MultipleShootingLayer)
  (; shooting_intervals, layer) = shooting
  ntuple(i -> LuxCore.initialparameters(rng, layer; tspan=shooting_intervals[i], shooting_layer=i != 1), length(shooting_intervals))
end

function LuxCore.initialstates(rng::Random.AbstractRNG, shooting::MultipleShootingLayer)
  (; shooting_intervals, layer) = shooting
  ntuple(i -> LuxCore.initialstates(rng, layer; tspan=shooting_intervals[i], shooting_layer=i != 1), length(shooting_intervals))
end

function (shooting::MultipleShootingLayer)(u0, ps, st) 
		(; layer) = shooting 
		ps = (
			merge(first(ps), (;u0 = u0)),
		Base.tail(ps)... 
		)
		shooting(nothing, ps, st)
end

function (shooting::MultipleShootingLayer)(::Nothing, ps, st)
  (; layer, ensemble_alg) = shooting
  shooting_problem = SingleShootingProblem(layer, first(ps), first(st))
  remaker = let ps = ps, st = st
    function (prob, i, repeat)
      remake(prob; ps=ps[i], st=st[i])
    end
  end
	ensemblesol = solve(EnsembleProblem(shooting_problem, prob_func=remaker, output_func=(sol, i) -> (first(sol), false)), DummySolve(), ensemble_alg; trajectories=length(st))
	Trajectory(shooting, ensemblesol), st
end

function Trajectory(::MultipleShootingLayer, sol::EnsembleSolution)
  (; u) = sol
  p = first(u).p
  sys = first(u).sys
  us = map(state_values, u)
  ts = map(current_time, u)
	tnew = reduce(vcat, map(i -> i == lastindex(ts) ? ts[i] : ts[i][1:end-1], eachindex(ts)))
	offsets = map(i->lastindex(us[i]) , eachindex(us[1:end-1])) |> cumsum
	shootings = map(i->last(us[i]) , eachindex(us[1:end-1]))
	unew = reduce(vcat, map(i -> i == lastindex(us) ? us[i] : us[i][1:end-1], eachindex(us)))
  Trajectory(sys, unew, p, tnew, shootings, offsets)
end
#=
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
struct MultipleShootingLayer{L,SI,E,B} <: LuxCore.AbstractLuxLayer
    "Collection of multiple SingleShootingLayer"
    layers::L
    "Collection of tspans for individual layers"
    shooting_intervals::SI
    "Ensemble method to solve EnsembleProblem"
    ensemble_alg::E
    "Bounds on the multiple shooting nodes"
    bounds_nodes::B
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


function build_optimal_control_solution(sol::EnsembleSolution)
	(; u ) = sol 
	p = first(u).p
	sys = first(u).sys 
	us = map(state_values, u) 
	ts = map(current_time, u)
	tnew = vcat(map(xi->xi[1:end-1], ts[1:end-1]), ts[end])
	offset = 0 
	shooting_points = map(us[1:end-1]) do ui 
		offset += lastindex(ui)
		offset, ui[end] 
	end
	unew = vcat(map(xi->xi[1:end-1], us[1:end-1]), us[end])
	Trajectory(sys, unew, p, tnew, shooting_points)  
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
                ensemble_alg = EnsembleSerial(), kwargs...)
    tspan = prob.tspan
    shooting_points = vcat(tspan..., shooting_points) |> unique! |> sort!
    shooting_intervals = [(t0,t1) for (t0,t1) in zip(shooting_points[1:end-1], shooting_points[2:end])]
    _tunable = vcat([tunable_ic], [collect(1:length(prob.u0)) for _ in 1:length(shooting_intervals)])
    layers = [SingleShootingLayer(remake(prob, tspan = tspani, kwargs...), alg, control_indices, restrict_controls(controls, tspani...);
                tunable_ic=_tunable[i], bounds_ic = (i == 1 ? (isempty(tunable_ic) ? nothing : bounds_ic) : bounds_nodes)) for (i, tspani) in enumerate(shooting_intervals)]

    MultipleShootingLayer{typeof(layers), typeof(shooting_intervals), typeof(ensemble_alg), typeof(bounds_nodes)}(layers, shooting_intervals, ensemble_alg, bounds_nodes)
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

struct Remaker{PS, ST}
	parameters::PS 
	states::ST 
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
    ensemblesol = solve(EnsembleProblem(prob, prob_func=remaker, output_func = (sol, i) -> (sol[1], false)),
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
=#
