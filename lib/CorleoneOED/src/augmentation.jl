function augment_system(mode::Val, prob::SciMLBase.AbstractDEProblem, alg::SciMLBase.AbstractDEAlgorithm;
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
  config = add_observed_equations(prob, config; control_indices, kwargs...)
  config = derive_sensitivity_equations(prob, alg, config; control_indices, kwargs...)
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

function compute_initial_F(prob::Union{ODEProblem,DAEProblem}, alg, config, params)
    (; vars, parameters, independent_vars, observed_jacobian) = config

    sol = solve(prob, alg, abstol=1e-10, reltol=1e-8)
    pred(p) = begin
        prob_ = remake(prob, p=p)
        Array(solve(prob_, alg, saveat=sol.t))[:]
    end

    hx = Symbolics.build_function(observed_jacobian, vars, parameters, only(independent_vars); expression=Val{false}, cse=true)[1]
    G = ForwardDiff.jacobian(pred, prob.p)

    nx = size(prob.u0, 1)
    F_tf_sens = sum(map(enumerate(diff(sol.t))) do (i,Δt)
        G_t = G[i*nx+1:(i+1)*nx, params]
        x = sol(sol.t[i+1])
        hx_ = hx(x,prob.p,sol.t[i+1])
        Δt * sum([(hx_[j:j,:] * G_t)' * (hx_[j:j,:] * G_t) for j in axes(hx_,1)])
    end)

    return F_tf_sens
end


function compute_svd_of_F(prob, alg, config, params; ns=nothing, threshold_singular_values=0.95, threshold_singular_vectors=0.1, kwargs...)
    F = compute_initial_F(prob, alg, config, params)
    svdF = svd(F)
    ns = !isnothing(ns) ? ns : findfirst(map(i-> sum((svdF.S.^2 / sum(abs2, svdF.S))[1:i]), eachindex(svdF.S)) .> threshold_singular_values)

    important_params = begin
        U = svdF.U[:,1:ns]
        U_important = abs.(U) .> threshold_singular_vectors
        map(x -> any(x), eachrow(U_important))
    end
    return svdF, ns, important_params
end

function derive_sensitivity_equations(prob, alg, config; params=Int64[], tunable_ic=Int64[], svd=false, ns = nothing, kwargs...)
  # TODO just switch this if we want to use the tunable_ics
  tunable_ic = empty(tunable_ic)
  (; differential_vars, vars, parameters, equations) = config
  svdF, ns, important_params = compute_svd_of_F(prob, alg, config, params; kwargs...)
  @info ns important_params

  psubset = svd ? parameters[params][important_params] : parameters[params]

  @info psubset
  np_considered = (svd ? ns : size(psubset, 1)) + size(tunable_ic, 1)
  nx = size(vars, 1)
  dG = Symbolics.variables(:dG, 1:nx, 1:np_considered)
  G = Symbolics.variables(:G, 1:nx, 1:np_considered)
  G0 = hcat(
    zeros(eltype(prob.u0), nx, size(np_considered, 1)),
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
  display(dfdp)
  sensitivities = dfdx * G + (svd ? dfdp * svdF.U[important_params,1:ns] : dfdp)
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
    G = sum(axes(observed_jacobian, 1)) do i
      observed_jacobian[i:i, :] * sensitivities
    end
    #G = observed_jacobian * sensitivities
    output_expression = G'G
  else
    output_expression = reduce(vcat, map(axes(observed_jacobian, 1)) do i
      observed_jacobian[i, :] * sensitivities
    end)
  end
  config = merge(config, (; vars=new_vars, differential_vars=new_differential_vars, equations=new_equations,
    observed=(;
      fisher=output_expression,
      sensitivities=sensitivities,
      observed,
    )))
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
    G = sum(axes(observed_jacobian, 1)) do i
      observed_jacobian[i:i, :] * sensitivities
    end
    output_expression = G'G
  else
    w = Symbolics.variables(:w, axes(observed_jacobian, 1))
    w = Symbolics.setdefaultval.(w, one(eltype(prob.u0)))
    output_expression = sum(enumerate(w)) do (i, wi)
      Gi = observed_jacobian[i:i, :] * sensitivities
      w[i] * Gi'Gi
    end
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
  config = merge(config, (; vars=new_vars, differential_vars=new_differential_vars, equations=new_equations,
    observed=(;
      fisher=fisher,
      sensitivities=sensitivities,
      observed,
    )))
  build_new_system(prob, config; control_indices, kwargs...)
end

function build_new_system(prob::ODEProblem, config; control_indices=Int64[], kwargs...)
  (; equations, vars, differential_vars, parameters, independent_vars, observed) = config
  (; observed_jacobian, observed, sensitivities) = config
  # Append the local information gain
  ex_local = reduce(vcat, map(axes(observed_jacobian, 1)) do i
    G = observed_jacobian[i:i, :] * sensitivities
	end)
	observed = merge(observed, (; local_weighted_sensitivity= Num.(ex_local)))
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
    fobs = getsym(layersys, Symbolics.SymbolicUtils.Code.toexpr.(ex))
    fobs
  end
  problem, obsfun
end

function build_new_system(prob::DAEProblem, config; control_indices=Int64[], kwargs...)
  (; equations, vars, differential_vars, parameters, independent_vars, observed) = config
  (; observed_jacobian, observed, sensitivities) = config
  # Append the local information gain
  ex_local = reduce(vcat, map(axes(observed_jacobian, 1)) do i
    G = observed_jacobian[i, :] * sensitivities
	end)
	observed = merge(observed, (; local_weighted_sensitivity= Num.(ex_local)))
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
