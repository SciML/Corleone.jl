function augment_system(mode::Val, prob::SciMLBase.AbstractDEProblem;
  control_indices=Int64[],
  kwargs...)
  sys = Corleone.retrieve_symbol_cache(prob, [])
  states = SymbolicIndexingInterface.variable_symbols(sys)
  sort!(states, by=Base.Fix1(SymbolicIndexingInterface.variable_index, sys))
  dstates = map(xi -> Symbol(:d, xi), states)
  ps = SymbolicIndexingInterface.parameter_symbols(sys)
  sort!(ps, by=Base.Fix1(SymbolicIndexingInterface.parameter_index, sys))
  ts = SymbolicIndexingInterface.independent_variable_symbols(sys)
  vars = Symbolics.variable.(states)
  vars = Symbolics.setdefaultval.(vars, vec(prob.u0))
  parameters = Symbolics.variable.(ps)
  p0, _ = SciMLStructures.canonicalize(SciMLStructures.Tunable(), prob.p)
  parameters = Symbolics.setdefaultval.(parameters, p0)
  config = (; symbolcache=sys, differential_vars=Symbolics.variable.(dstates),
    vars=vars,
    parameters=parameters,
    independent_vars=Symbolics.variable.(ts)
  )
  config = symbolify_equations(prob, config; control_indices, kwargs...)
  config = derive_sensitivity_equations(prob, config; control_indices, kwargs...)
  config = add_observed_equations(prob, config; control_indices, kwargs...)
  finalize_config(mode, prob, config; control_indices, kwargs...)
end

function symbolify_equations(prob::SciMLBase.AbstractDEProblem, config; kwargs...)
  (; differential_vars, vars, parameters, independent_vars) = config
  f = prob.f.f
  eqs = if SciMLBase.isinplace(prob)
    out = zero(vars)
    f(out, vars, parameters, only(independent_vars))
    out
  else
    f(vars, parameters, only(independent_vars))
  end
  merge(config, (; equations=eqs))
end

function symbolify_equations(prob::SciMLBase.DAEProblem, config; kwargs...)
  (; differential_vars, vars, parameters, independent_vars) = config
  f = prob.f.f
  eqs = if SciMLBase.isinplace(prob)
    out = zero(vars)
    f(out, differential_vars, vars, parameters, only(independent_vars))
    out
  else
    f(differential_vars, vars, parameters, only(independent_vars))
  end
  merge(config, (; equations=eqs))
end

function derive_sensitivity_equations(prob, config; params=Int64[], tunable_ic=Int64[], kwargs...)
  # TODO just switch this if we want to use the tunable_ics 
  tunable_ic = empty(tunable_ic)
  (; symbolcache, differential_vars, vars, parameters, independent_vars, equations) = config
  psubset = parameters[params]
  np_considered = size(psubset, 1) + size(tunable_ic, 1)
  nx = size(vars, 1)

  dG = Symbolics.variables(:dG, 1:nx, 1:np_considered)
  G = Symbolics.variables(:G, 1:nx, 1:np_considered)
  G0 = hcat(
    zeros(eltype(prob.u0), nx, size(psubset, 1)),
    [(i == tunable_ic[j]) + zero(eltype(prob.u0)) for i in 1:nx, j in eachindex(tunable_ic)]
  )
  G = Symbolics.setdefaultval.(G, G0)
  dfdx = Symbolics.jacobian(equations, vars)
  dfddx = Symbolics.jacobian(equations, differential_vars)
  dfdp = Symbolics.jacobian(equations, psubset)
  if !isempty(tunable_ic)
    dfdpextra = [(i == tunable_ic[j]) + zero(eltype(prob.u0)) for i in 1:nx, j in eachindex(tunable_ic)]
    dfdp = hcat(dfdp, dfdpextra)
  end
  sensitivities = dfdx * G + dfdp
  if isa(prob, SciMLBase.DAEProblem)
    sensitivities .+= dfddx * dG
  end
  merge(config, (; sensitivities=G, differential_sensitivities=dG, sensitivity_equations=sensitivities))
end

function add_observed_equations(prob, config; observed=(u, p, t) -> u, kwargs...)
  (; symbolcache, differential_vars, vars, parameters, independent_vars, equations) = config
  obs = observed(vars, parameters, only(independent_vars))
  dobsdx = Symbolics.jacobian(obs, vars)
  merge(config, (; observed=obs, observed_jacobian=dobsdx))
end

finalize_config(::Any, args...; kwargs...) = throw(ErrorException("The OED cannot be derived based on the given information. This should never happen. Please open up an issue."))

# Just the sensitivities, no weigthing, no
function finalize_config(::T, prob, config; kwargs...) where {T<:Union{Val{:Discrete},Val{:DiscreteSampled}}}
  (; symbolcache, differential_vars, vars, parameters, independent_vars, equations) = config
  (; sensitivities, differential_sensitivities, sensitivity_equations) = config
  (; observed_jacobian, observed) = config
  new_vars = vcat(vars, vec(sensitivities))
  new_differential_vars = vcat(differential_vars, vec(differential_sensitivities))
  new_equations = vcat(equations, vec(sensitivity_equations))
  # We build the output expression 
  if T == Val{:Discrete}
    G = sum(axes(observed_jacobian,1)) do i
      observed_jacobian[i:i, :] * sensitivities
    end
    #G = observed_jacobian * sensitivities
    output_expression = G'G
  else
    output_expression = reduce(vcat, map(axes(observed_jacobian, 1)) do i
      observed_jacobian[i, :] * sensitivities
    end)
  end
  config = merge(config, (; vars=new_vars, differential_vars=new_differential_vars, equations=new_equations, observed=(; fisher=output_expression, observed)))
  build_new_system(prob, config; kwargs...)
end

function finalize_config(::T, prob, config; control_indices=Int64[], kwargs...) where {T<:Union{Val{:Continuous},Val{:ContinuousSampled}}}
  (; symbolcache, differential_vars, vars, parameters, independent_vars, equations) = config
  (; sensitivities, differential_sensitivities, sensitivity_equations) = config
  (; observed_jacobian, observed) = config
  n = size(sensitivities, 2)
  selector = triu(trues(n, n))
  F = Symbolics.variables(:F, 1:n, 1:n)
  F = Symbolics.setdefaultval.(F, zero(eltype(prob.u0)))
  dF = Symbolics.variables(:dF, 1:n, 1:n)
  # We build the output expression 
  if T != Val{:ContinuousSampled}
    G = sum(axes(observed_jacobian,1)) do i
      observed_jacobian[i:i, :] * sensitivities
    end
    output_expression = G'G
  else
    w = Symbolics.variables(:w, axes(observed_jacobian, 1))
    w = Symbolics.setdefaultval.(w, one(eltype(prob.u0)))
    G = sum(enumerate(w)) do (i, wi)
      wi * observed_jacobian[i:i, :] * sensitivities
    end
    output_expression = G'G
    idx = axes(w, 1) .+ size(parameters, 1)
    append!(parameters, w)
    append!(control_indices, idx)
  end
  output_expression = vec(output_expression[selector])
  fisher = [selector[i, j] ? F[i, j] : F[j, i] for i in 1:n, j in 1:n]
  F = F[selector]
  dF = dF[selector]
  if isa(prob, DAEProblem)
    output_expression = vec(dF) .- output_expression
  end
  new_vars = vcat(vars, vec(sensitivities), vec(F))
  new_differential_vars = vcat(differential_vars, vec(differential_sensitivities), vec(dF))
  new_equations = vcat(equations, vec(sensitivity_equations), vec(output_expression))
  config = merge(config, (; vars=new_vars, differential_vars=new_differential_vars, equations=new_equations, observed=(; fisher=fisher, observed)))
  build_new_system(prob, config; control_indices, kwargs...)
end

function build_new_system(prob::ODEProblem, config; control_indices=Int64[], kwargs...)
  (; equations, vars, differential_vars, parameters, independent_vars, observed) = config
  @info equations
  IIP = SciMLBase.isinplace(prob)
  foop, fiip = Symbolics.build_function(equations, vars, parameters, only(independent_vars); expression=Val{false}, cse=true)
  u0 = Symbolics.getdefaultval.(vars)
  p0 = Symbolics.getdefaultval.(parameters)
  defaults = Dict(vcat(Symbol.(vars), Symbol.(parameters)) .=> vcat(u0, p0))
  newsys = SymbolCache(
    Symbol.(vars), Symbol.(parameters), independent_vars;
    defaults=defaults
  )
  # Note: This is different 
  fnew = ODEFunction(IIP ? fiip : foop, sys=newsys)
  problem = remake(prob, f=fnew, u0=u0, p=p0)
  layersys = Corleone.retrieve_symbol_cache(problem, control_indices)
  obsfun = map(observed) do ex
    getsym(layersys, Symbolics.SymbolicUtils.Code.toexpr.(ex))
  end
  problem, obsfun
end

function build_new_system(prob::DAEProblem, config; control_indices=Int64[], kwargs...)
  (; equations, vars, differential_vars, parameters, independent_vars, observed) = config
  @info equations
  IIP = SciMLBase.isinplace(prob)
  foop, fiip = Symbolics.build_function(equations, differential_vars, vars, parameters, only(independent_vars); expression=Val{false}, cse=true)
  u0 = Symbolics.getdefaultval.(vars)
  p0 = Symbolics.getdefaultval.(parameters)
  du0 = vcat(prob.du0, zeros(eltype(u0), size(u0, 1) - size(prob.du0, 1)))
  defaults = Dict(vcat(Symbol.(vars), Symbol.(parameters)) .=> vcat(u0, p0))
  newsys = SymbolCache(
    Symbol.(vars), Symbol.(parameters), independent_vars;
    defaults=defaults
  )
  fnew = DAEFunction(IIP ? fiip : foop, sys=newsys)
  problem = remake(prob, f=fnew, du0=du0, u0=u0, p=p0)
  layersys = Corleone.retrieve_symbol_cache(problem, control_indices)
  obsfun = map(observed) do ex
    getsym(layersys, Symbolics.SymbolicUtils.Code.toexpr.(ex))
  end
  problem, obsfun
end


struct OEDLayer{DISCRETE,SAMPLED,FIXED,L,O} <: LuxCore.AbstractLuxWrapperLayer{:layer}
  "The underlying layer"
  layer::L
  "The observed functions"
  observed::O
  "The sampling indices"
  sampling_indices::Vector{Int64}
end

function OEDLayer{DISCRETE}(layer::L, args...; measurements=[], kwargs...) where {DISCRETE,L}

  (; problem, algorithm, controls, control_indices, tunable_ic, bounds_ic, state_initialization, bounds_p, parameter_initialization) = layer

  SAMPLED = !isempty(measurements)
  mode = DISCRETE ? (SAMPLED ? Val{:DiscreteSampled}() : Val{:Discrete}()) : (SAMPLED ? Val{:ContinuousSampled}() : Val{:Continuous}())
  @info mode
  p_length = length(problem.p)
  samplings = SAMPLED ? collect(eachindex(measurements)) : Int64[]
  ctrls = vcat(collect(control_indices .=> controls), samplings .+ p_length .=> measurements)
  samplings = samplings .+ length(controls)

  newproblem, observed = augment_system(mode, problem;
    tunable_ic=copy(tunable_ic),
    control_indices=copy(control_indices),
    kwargs...)

  # Replace the saveat with the sampling times 
  saveats = if SAMPLED
    ts = reduce(vcat, Corleone.get_timegrid.(measurements))
    unique!(sort!(ts))
  else
    collect(problem.tspan)
  end
  newproblem = remake(newproblem, saveat=saveats)

  lb, ub = copy.(bounds_ic)
  for i in eachindex(newproblem.u0)
    i <= lastindex(problem.u0) && continue
    push!(lb, zero(eltype(newproblem.u0)))
    push!(ub, zero(eltype(newproblem.u0)))
  end
  newlayer = SingleShootingLayer(
    newproblem, algorithm; controls=ctrls, tunable_ic=copy(tunable_ic), bounds_ic=(lb, ub), state_initialization, bounds_p, parameter_initialization
  )

  OEDLayer{DISCRETE,SAMPLED,LuxCore.parameterlength(layer) == 0,typeof(newlayer),typeof(observed)}(newlayer, observed, samplings)
end


Corleone.get_bounds(oed::OEDLayer; kwargs...) = Corleone.get_bounds(oed.layer; kwargs...) 

struct WeightedObservation
  grid::Vector{Vector{Int64}}
end

function (w::WeightedObservation)(controls::AbstractVector{T}, i::Int64, G::AbstractArray) where {T}
  psub = [iszero(i) ? zero(T) : controls[i] for i in w.grid[i]]
  G = psub .* G
  G'G
end

function (w::WeightedObservation)(controls::AbstractVector{T}, G::AbstractVector{<:AbstractArray}) where {T}
  sum(eachindex(G)) do i
    w(controls, i, G[i])
  end
end

# This is the only case where we need to sample the trajectory
function LuxCore.initialstates(rng::Random.AbstractRNG, oed::OEDLayer{true,true})
  (; layer, sampling_indices) = oed
  (; problem, controls, control_indices) = layer
  st = LuxCore.initialstates(rng, layer)
  # Our goal is to build a weigthing matrix similar to the indexgrid 
  grids = Corleone.get_timegrid.(controls)
  overall_grid = vcat(reduce(vcat, grids), collect(problem.tspan))
	unique!(sort!(overall_grid))
  observed_grid = map(grids[sampling_indices]) do grid
    unique!(sort!(grid))
    findall(∈(grid), overall_grid)
  end
  measurement_indices = Corleone.build_index_grid(controls...; problem.tspan, subdivide=100)
  measurement_indices = map(eachrow(measurement_indices[sampling_indices, :])) do mi
    unique(mi)
  end
  # Lets order this by time 
  weighting_grid = map(eachindex(overall_grid)) do i
    map(eachindex(observed_grid)) do j
      id = findfirst(i .== observed_grid[j])
      isnothing(id) && return 0
      measurement_indices[j][id]
    end
  end
  merge(st, (; observation_grid=WeightedObservation(weighting_grid)))
end



__fisher_information(oed::OEDLayer, traj::Trajectory) = oed.observed.fisher(traj)

fisher_information(oed::OEDLayer, x, ps, st::NamedTuple) = begin
  traj, st = oed(x, ps, st)
  sum(__fisher_information(oed, traj)), st
end

# Continuous ALWAYS last FIM 
fisher_information(oed::OEDLayer{false}, x, ps, st::NamedTuple) = begin
  traj, st = oed(x, ps, st)
	last(__fisher_information(oed, traj)), st
end

# DISCRETE and SAMPLING -> weighted sum 
fisher_information(oed::OEDLayer{true,true}, x, ps, st::NamedTuple) = begin
  (; sampling_indices, layer) = oed
  (; observation_grid) = st
  traj, st = oed(x, ps, st)
  Gs = __fisher_information(oed, traj)
  observation_grid(ps.controls, Gs), st
end

# DISCRETE -> SUM
fisher_information(oed::OEDLayer{true,false}, x, ps, st::NamedTuple) = begin
  (; sampling_indices, layer) = oed
  (; observation_grid) = st
  traj, st = oed(x, ps, st)
	sum(__fisher_information(oed, traj)), st
end

observed_equations(oed::OEDLayer, traj::Trajectory) = oed.observed.observed(traj)

observed_equations(oed::OEDLayer, x, ps, st::NamedTuple) = begin
  traj, st = oed(x, ps, st)
  observed_equations(oed, traj), st
end

