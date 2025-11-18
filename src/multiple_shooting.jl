struct MultipleShootingLayer{L,I,E} <: LuxCore.AbstractLuxWrapperLayer{:layer}
  "The original layer"
  layer::L
  "The shooting intervals"
  shooting_intervals::I
  "The ensemble algorithm"
  ensemble_alg::E
end

function Base.show(io::IO, layer::MultipleShootingLayer)
  type_color, no_color = SciMLBase.get_colorizers(io)
  print(io,
    type_color, "MultipleShootingLayer ",
    no_color, "with $(length(layer.shooting_intervals)) shooting intervals and $(length(get_controls(layer.layer))) controls.\n")
  print(io, "Underlying problem: ")
  print(io, layer.layer)
end


MultipleShootingLayer(prob, alg, tpoints::AbstractVector; kwargs...) = MultipleShootingLayer(prob, alg, tpoints...; kwargs...)

function MultipleShootingLayer(prob::SciMLBase.AbstractDEProblem, alg::SciMLBase.DEAlgorithm, tpoints::Real...; ensemble_alg = EnsembleSerial(), kwargs...)
  layer = SingleShootingLayer(prob, alg; kwargs...)
  MultipleShootingLayer(layer, tpoints...; ensemble_alg, kwargs...)
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
  names = ntuple(i -> Symbol(:interval, "_", i), length(shooting_intervals))
  vals = ntuple(i -> LuxCore.initialparameters(rng, layer; tspan=shooting_intervals[i], shooting_layer=i != 1), length(shooting_intervals))
  NamedTuple{names}(vals)
end

function LuxCore.initialstates(rng::Random.AbstractRNG, shooting::MultipleShootingLayer)
  (; shooting_intervals, layer) = shooting
  names = ntuple(i -> Symbol(:interval, "_", i), length(shooting_intervals))
  vals = ntuple(i -> LuxCore.initialstates(rng, layer; tspan=shooting_intervals[i], shooting_layer=i != 1), length(shooting_intervals))
  NamedTuple{names}(vals)
end


function (shooting::MultipleShootingLayer)(u0, ps, st::NamedTuple{fields}) where {fields}
  (; layer, ensemble_alg) = shooting
  ret = Corleone._parallel_solve(ensemble_alg, layer, u0, ps, st)
	u = first.(ret) 
	sts = NamedTuple{fields}(last.(ret))
  Trajectory(u, sts), sts
end

function Trajectory(u::AbstractVector, sts)
  size(u, 1) == 1 && return only(u)
  p = first(u).p
  sys = first(u).sys
  us = map(state_values, u)
  ts = map(current_time, u)
  tnew = reduce(vcat, map(i -> i == lastindex(ts) ? ts[i] : ts[i][1:end-1], eachindex(ts)))
  offsets = map(i -> lastindex(us[i]), eachindex(us[1:end-1])) |> cumsum
  shootings = map(eachindex(us[1:end-1])) do i
    uprev = us[i]
    unext = us[i+1]
    idx = sts[i+1].shooting_indices
    vcat(last(uprev)[idx] .- first(unext)[idx], u[i].p .- u[i+1].p)
  end
  unew = reduce(vcat, map(i -> i == lastindex(us) ? us[i] : us[i][1:end-1], eachindex(us)))
  Trajectory(sys, unew, p, tnew, shootings, offsets), sts
end

"""
    get_block_structure(layer)

Compute the block structure of the hessian of the Lagrangian of an optimal control problem
as specified via the `shooting_intervals` of the `MultipleShootingLayer`.
"""
function get_block_structure(mslayer::MultipleShootingLayer)
  (; layer, shooting_intervals) = mslayer
	ps_lengths = collect(map(enumerate(shooting_intervals)) do (i,tspan)
    LuxCore.parameterlength(layer, tspan=tspan, shooting_layer = i > 1)
  end)
  vcat(0, cumsum(ps_lengths))
end

function get_bounds(mslayer::MultipleShootingLayer)
  (; layer, shooting_intervals) = mslayer
  names = ntuple(i -> Symbol(:interval, "_", i), length(shooting_intervals))
  bounds = map(shooting_intervals) do tspan
    get_bounds(layer; tspan=tspan)
  end
  NamedTuple{names}(first.(bounds)), NamedTuple{names}(last.(bounds))
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
