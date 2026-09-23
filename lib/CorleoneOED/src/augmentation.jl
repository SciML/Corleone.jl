function augment_system(
        prob::SciMLBase.AbstractDEProblem, alg::SciMLBase.AbstractDEAlgorithm;
        control_indices = Int64[], svd = false, fixed::Bool = false,
        kwargs...
    )
    sys = prob.f.sys
    states = SymbolicIndexingInterface.variable_symbols(sys)
    sort!(states, by = Base.Fix1(SymbolicIndexingInterface.variable_index, sys))
    dstates = map(xi -> Symbol(:d, xi), states)
    ps = SymbolicIndexingInterface.parameter_symbols(sys)
    sort!(ps, by = Base.Fix1(SymbolicIndexingInterface.parameter_index, sys))
    ts = SymbolicIndexingInterface.independent_variable_symbols(sys)
    vars = Symbolics.variable.(states)
    vars = Symbolics.setdefaultval.(vars, vec(prob.u0))
    parameters = Symbolics.variable.(ps)
    p0, _ = SciMLStructures.canonicalize(SciMLStructures.Tunable(), prob.p)
    parameters = Symbolics.setdefaultval.(parameters, p0)
    config = (;
        symbolcache = sys, differential_vars = Symbolics.variable.(dstates),
        vars = vars,
        parameters = parameters,
        independent_vars = Symbolics.variable.(ts),
    )
    config = symbolify_equations(prob, config; control_indices, kwargs...)
    config = add_observed_equations(prob, config; control_indices, kwargs...)
    if svd
        config = derive_sensitivity_equations_svd(prob, alg, config; control_indices, kwargs...)
    else
        config = derive_sensitivity_equations(prob, alg, config; control_indices, kwargs...)
    end
    return finalize_config(prob, config; control_indices, kwargs...)
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
    return merge(config, (; equations = eqs))
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
    return merge(config, (; equations = eqs))
end

function compute_initial_F(prob::Union{ODEProblem, DAEProblem}, alg, config, params)
    (; vars, parameters, independent_vars, observed_jacobian) = config

    sol = solve(prob, alg, abstol = 1.0e-10, reltol = 1.0e-8)
    pred(p) = begin
        prob_ = remake(prob, p = p)
        Array(solve(prob_, alg, saveat = sol.t))[:]
    end

    hx = Symbolics.build_function(observed_jacobian, vars, parameters, only(independent_vars); expression = Val{false}, cse = true)[1]
    G = ForwardDiff.jacobian(pred, prob.p)

    nx = size(prob.u0, 1)
    F_tf_sens = sum(
        map(enumerate(diff(sol.t))) do (i, Δt)
            G_t = G[(i * nx + 1):((i + 1) * nx), params]
            x = sol(sol.t[i + 1])
            hx_ = hx(x, prob.p, sol.t[i + 1])
            Δt * sum([(hx_[j:j, :] * G_t)' * (hx_[j:j, :] * G_t) for j in axes(hx_, 1)])
        end
    )

    return F_tf_sens
end

function compute_svd_of_F(prob, alg, config, params; ns = nothing, threshold_singular_values = 0.95, threshold_singular_vectors = 0.1, kwargs...)
    F = compute_initial_F(prob, alg, config, params)
    svdF = svd(F)

    ns = !isnothing(ns) ? ns : findfirst(map(i -> sum((svdF.S .^ 2 / sum(abs2, svdF.S))[1:i]), eachindex(svdF.S)) .> threshold_singular_values)

    important_params = begin
        U = svdF.U[:, 1:ns]
        U_important = abs.(U) .> threshold_singular_vectors
        map(x -> any(x), eachrow(U_important))
    end
    return svdF, ns, important_params
end

function derive_sensitivity_equations_svd(prob, alg, config; 
        params = Int64[], tunable_ic = Int64[], kwargs...)
    # TODO just switch this if we want to use the tunable_ics
    tunable_ic = empty(tunable_ic)
    (; differential_vars, vars, parameters, equations) = config

    svdF, ns, important_params = compute_svd_of_F(prob, alg, config, params; kwargs...)

    psubset = parameters[params[important_params]]

    np_considered = ns + size(tunable_ic, 1)
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
    sensitivities = dfdx * G + dfdp * svdF.U[important_params, 1:ns]
    if isa(prob, SciMLBase.DAEProblem)
        sensitivities .+= dfddx * dG
    end
    return merge(config, (; sensitivities = G, differential_sensitivities = dG, sensitivity_equations = sensitivities))
end

function select_subset_params(parameters, params::AbstractVector{<:Int})
    return parameters[params]
end

function select_subset_params(parameters, params::AbstractVector{<:Symbol})
    @assert all([any(Base.Fix1(isequal, Symbolics.variable(p)).(parameters)) for p in params]) "Augmentation: Some of the selected parameters are not in the symbol cache!"
    idxs = map(p -> argmax(Base.Fix1(isequal, Symbolics.variable(p)).(parameters)), params)
    return parameters[idxs]
end


function derive_sensitivity_equations(prob, alg, config; 
            params::Union{AbstractVector{<:Int}, AbstractVector{<:Symbol}} = Int64[],  
            tunable_ic = Int64[], kwargs...)

    # TODO just switch this if we want to use the tunable_ics
    tunable_ic = empty(tunable_ic)
    (; differential_vars, vars, parameters, equations) = config

    psubset = select_subset_params(parameters, params)

    np_considered = size(psubset, 1) + size(tunable_ic, 1)
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
    sensitivities = dfdx * G + dfdp
    if isa(prob, SciMLBase.DAEProblem)
        sensitivities .+= dfddx * dG
    end
    return merge(config, (; sensitivities = G, differential_sensitivities = dG, sensitivity_equations = sensitivities))
end

function add_observed_equations(prob, config; continuous_measurements = ContinuousMeasurement[], 
            discrete_measurements = DiscreteMeasurement[], kwargs...)
    (; symbolcache, differential_vars, vars, parameters, independent_vars, equations) = config

    obs_cont = reduce(vcat, map(obs -> obs.observed(vars, parameters, only(independent_vars)), continuous_measurements); init = Num[])
    dobs_cont_dx = Symbolics.jacobian(obs_cont, vars)
    config = merge(config, (; observed_continuous = obs_cont, observed_continuous_jacobian = dobs_cont_dx))
    obs_disc = reduce(vcat, map(obs -> obs.observed(vars, parameters, only(independent_vars)), discrete_measurements), init = Num[])
    dobs_disc_dx = Symbolics.jacobian(obs_disc, vars)
    merge(config, (; observed_discrete = obs_disc, observed_discrete_jacobian = dobs_disc_dx))
end

finalize_config(::Any, args...; kwargs...) = throw(ErrorException("The OED cannot be derived based on the given information. This should never happen. Please open up an issue."))

# Continuous, non fixed version
function finalize_config(prob, config; control_indices = Int64[], continuous_measurements = ContinuousMeasurement[],
            discrete_measurements = DiscreteMeasurement[], kwargs...)
    (; symbolcache, differential_vars, vars, parameters, independent_vars, equations) = config
    (; sensitivities, differential_sensitivities, sensitivity_equations) = config
    (; observed_continuous_jacobian, observed_continuous) = config
    (; observed_discrete_jacobian, observed_discrete) = config
    
    n = size(sensitivities, 2)

    selector = triu(trues(n, n))
    F = Symbolics.variables(:F, 1:n, 1:n)
    F = Symbolics.setdefaultval.(F, zero(eltype(prob.u0)))
    dF = Symbolics.variables(:dF, 1:n, 1:n)
    # We build the output expression

    G_disc = observed_discrete_jacobian * sensitivities

    disc_names = [x.id for x in discrete_measurements]
    cont_names = [x.id for x in continuous_measurements]

    w_disc = Symbolics.variable.(disc_names)
    w_disc = Symbolics.setdefaultval.(w_disc, one(eltype(prob.u0)))

    w_cont = Symbolics.variable.(cont_names)
    w_cont = Symbolics.setdefaultval.(w_cont, one(eltype(prob.u0)))

    F_cont = !isempty(w_cont) ? sum(enumerate(w_cont)) do (i, wi)
        Gi = observed_continuous_jacobian[i:i, :] * sensitivities
        wi * Gi'Gi
    end : zero.(F)
    idx_disc = axes(w_disc, 1) .+ size(parameters, 1)
    idx_cont = axes(w_cont, 1) .+ size(parameters, 1) .+ size(observed_discrete_jacobian, 1)
    append!(parameters, w_disc)
    append!(parameters, w_cont)
    append!(control_indices, idx_disc)
    append!(control_indices, idx_cont)

    F_eqs = vec(F_cont[selector])
    fisher = [selector[i, j] ? F[i, j] : F[j, i] for i in 1:n, j in 1:n]
    F = F[selector]
    dF = dF[selector]
    if isa(prob, DAEProblem)
        F_eqs = vec(dF) .- F_eqs
    end
    new_vars = vcat(vars, vec(sensitivities), vec(F))
    new_differential_vars = vcat(differential_vars, vec(differential_sensitivities), vec(dF))
    new_equations = vcat(equations, vec(sensitivity_equations), vec(F_eqs))
    config = merge(
        config, (;
            vars = new_vars, differential_vars = new_differential_vars, equations = new_equations,
            observed = (;
                fisher = fisher,
                sensitivities = sensitivities,
                hx_G_discrete = G_disc,
                observed_continuous = observed_continuous,
                observed_discrete = observed_discrete
            ),
        )
    )
    return build_new_system(prob, config; control_indices, kwargs...)
end

function build_new_system(prob::ODEProblem, config; control_indices = Int64[], kwargs...)
    (; equations, vars, differential_vars, parameters, independent_vars, observed) = config
    (; observed_continuous_jacobian, observed_discrete_jacobian, sensitivities) = config
    # Append the local information gain
    ex_local_cont = observed_continuous_jacobian * sensitivities
    ex_local_disc = observed_discrete_jacobian * sensitivities
    observed = merge(observed, (; local_information_gain = Num.(vcat(ex_local_cont, ex_local_disc))))
    IIP = SciMLBase.isinplace(prob)
    foop, fiip = Symbolics.build_function(equations, vars, parameters, only(independent_vars); expression = Val{false}, cse = true)
    u0 = Symbolics.getdefaultval.(vars)
    p0 = Symbolics.getdefaultval.(parameters)
    defaults = Dict(vcat(Symbol.(vars), Symbol.(parameters)) .=> vcat(u0, p0))
    newsys = SymbolCache(
        Symbol.(vars), Symbol.(parameters), independent_vars;
        defaults = defaults
    )
    # Note: This is different
    fnew = ODEFunction(IIP ? fiip : foop, sys = newsys)
    problem = remake(prob, f = fnew, u0 = u0, p = p0)

    obsfun = map(observed) do ex
        fobs = getsym(problem, Symbolics.SymbolicUtils.Code.toexpr.(ex))
        fobs
    end
    return problem, obsfun
end

function build_new_system(prob::DAEProblem, config; control_indices = Int64[], kwargs...)
    (; equations, vars, differential_vars, parameters, independent_vars, observed) = config
    (; observed_continuous_jacobian, observed_discrete_jacobian, sensitivities) = config
    # Append the local information gain
    ex_local_cont = observed_continuous_jacobian * sensitivities
    ex_local_disc = observed_discrete_jacobian * sensitivities
    observed = merge(observed, (; local_information_gain = Num.(vcat(ex_local_cont, ex_local_disc))))
    IIP = SciMLBase.isinplace(prob)
    foop, fiip = Symbolics.build_function(equations, differential_vars, vars, parameters, only(independent_vars); expression = Val{false}, cse = true)
    u0 = Symbolics.getdefaultval.(vars)
    p0 = Symbolics.getdefaultval.(parameters)
    du0 = vcat(prob.du0, zeros(eltype(u0), size(u0, 1) - size(prob.du0, 1)))

    _du0 = foop(du0, u0, prob.p, 0.0)
    du0 = vcat(prob.du0, _du0[(size(prob.du0, 1) + 1):end])

    diff_vars = vcat(prob.differential_vars, ones(Bool, size(u0, 1) - size(prob.differential_vars, 1)))
    defaults = Dict(vcat(Symbol.(vars), Symbol.(parameters)) .=> vcat(u0, p0))
    newsys = SymbolCache(
        Symbol.(vars), Symbol.(parameters), independent_vars;
        defaults = defaults
    )
    fnew = DAEFunction(IIP ? fiip : foop, sys = newsys)

    problem = remake(prob, f = fnew, du0 = du0, u0 = u0, p = p0, differential_vars = diff_vars)
    obsfun = map(observed) do ex
        fobs = getsym(problem, Symbolics.SymbolicUtils.Code.toexpr.(ex))
        fobs
    end
    return problem, obsfun
end
