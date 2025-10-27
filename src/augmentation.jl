"""
$(METHODLIST)

Augments the `prob` with the equations for the sensitivities of `params` and the Fisher
information matrix. This method is used in cases when the states and sensitivities are
not constant, i.e., there are other controls or initial conditions to be optimized jointly
with the sampling decisions.
Returns either an ODEProblem or a DAEProblem, depending on the type of the original problem.
"""
function augment_dynamics_full(prob::SciMLBase.AbstractDEProblem;
            tspan=prob.tspan, control_indices = Int64[],
            params = setdiff(eachindex(prob.p), control_indices), observed = (u,p,t) -> u[eachindex(prob.u0)])

    is_dae = isa(prob, SciMLBase.DAEProblem)
    u0 = prob.u0
    nx, np, nc, np_considered = length(prob.u0), length(prob.p), length(control_indices), length(params)

    iip = SciMLBase.isinplace(prob)
    _dx = Symbolics.variables(:dx, 1:nx)
    _x = Symbolics.variables(:x, 1:nx)
    _p = Symbolics.variables(:p, 1:np)
    _t = Symbolics.variable(:t)

    _dynamics = begin
        if iip
            if !is_dae
                prob.f.f(_dx, _x, _p, _t)
                _dx
            else
                out = Symbolics.variables(:out, 1:nx)
                prob.f.f(out, _dx, _x, _p, _t)
                out
            end
        else
            if !is_dae
                prob.f.f(_x, _p, _t)
            else
                prob.f.f(_dx, _x, _p, _t)
            end
        end
    end

    _dG = Symbolics.variables(:G, 1:nx, 1:np_considered)
    _G = Symbolics.variables(:G, 1:nx, 1:np_considered)
    _dF = Symbolics.variables(:F, 1:np_considered, 1:np_considered)
    _F = Symbolics.variables(:F, 1:np_considered, 1:np_considered)

    dfdx = Symbolics.jacobian(_dynamics, _x)
    dfddx = Symbolics.jacobian(_dynamics, _dx)
    dfdp = Symbolics.jacobian(_dynamics, _p[params])

    dGdt = is_dae ?  dfdp + dfdx * _G  + dfddx * _dG : dfdp + dfdx * _G
    _obs = observed(_x, _p, _t)
    _w = Symbolics.variables(:w, 1:length(_obs))
    dobsdx = Symbolics.jacobian(_obs, _x)

    upper_triangle = triu(trues(np_considered, np_considered))
    dFdt = sum(map(1:length(_obs)) do i
            _w[i] * (dobsdx[i:i,:] * _G)' * (dobsdx[i:i,:] * _G)
    end)[upper_triangle]
    dFdt = is_dae ? _dF[upper_triangle] .- dFdt : dFdt

    differential_variables_ = vcat(_dx, _dG[:], _dF[upper_triangle])
    variables_ = vcat(_x, _G[:], _F[upper_triangle])
    parameters_ = vcat(_p,_w)
    expressions_ = vcat(_dynamics, dGdt[:], dFdt)

    iip_idx = iip ? 2 : 1
    aug_fun = begin
        if !is_dae
            Symbolics.build_function(expressions_, variables_, parameters_, _t; expression=Val{false}, cse=true)[iip_idx]
        else
            Symbolics.build_function(expressions_, differential_variables_, variables_, parameters_, _t; expression=Val{false}, cse=true)[iip_idx]
        end
    end
    #dfun = eval(aug_fun)
    aug_u0 = vcat(u0, zeros(length(variables_) - length(u0)))
    aug_p = vcat(prob.p, ones(length(_w)))

    scache = SymbolCache(variables_, parameters_, [_t])
    dfun = is_dae ? DAEFunction(aug_fun, sys=scache) : ODEFunction(aug_fun, sys=scache)

    !is_dae && return ODEProblem{iip}(dfun, aug_u0, tspan, aug_p; prob.kwargs...)

    aug_du0 = vcat(prob.du0, zeros(length(differential_variables_) - length(prob.du0)))
    aug_diff_vars = isnothing(prob.differential_vars) ? nothing : vcat(prob.differential_vars, trues(length(differential_variables_) - length(prob.du0)))
    return DAEProblem{iip}(dfun, aug_du0, aug_u0, tspan, aug_p; differential_vars = aug_diff_vars, prob.kwargs...)
end

"""
$(METHODLIST)

Augments the `prob` with the equations for the sensitivities of `params` and the unweighted
contributions of each measurement function to the Fisher information matrix.
This method is used in cases when the states and sensitivities are constant.
Thus, unweighted contributions of all measurement functions to the FIM need to be
added for a simplified optimization problem later on.
Returns either an ODEProblem or a DAEProblem, depending on the type of the original problem.

"""
function augment_dynamics_unweighted_fisher(prob::SciMLBase.AbstractDEProblem;
            tspan=prob.tspan, control_indices = Int64[],
            params = setdiff(eachindex(prob.p), control_indices), observed = (u,p,t) -> u[eachindex(prob.u0)])

    is_dae = isa(prob, SciMLBase.DAEProblem)
    u0 = prob.u0
    nx, np, nc, np_considered = length(prob.u0), length(prob.p), length(control_indices), length(params)

    iip = SciMLBase.isinplace(prob)
    _dx = Symbolics.variables(:dx, 1:nx)
    _x = Symbolics.variables(:x, 1:nx)
    _p = Symbolics.variables(:p, 1:np_considered)
    full_p = Symbolics.variables(:p, 1:np)
    _t = Symbolics.variable(:t)

    counter_p = 0
    p_vector = map(eachindex(prob.p)) do i
        if i in params
            counter_p +=1
            _p[counter_p]
        else
            prob.p[i]
        end
    end
    p_vector .= prob.p
    p_vector[params] .= _p
    _dynamics = begin
        if iip
            out = Symbolics.variables(:out, 1:nx)
            if !is_dae
                prob.f.f(out, _x, p_vector, _t)
                out
            else
                prob.f.f(out, _dx, _x, p_vector, _t)
                out
            end
        else
            if !is_dae
                prob.f.f(_x, p_vector, _t)
            else
                prob.f.f(_dx, _x, p_vector, _t)
            end
        end
    end

    _dG = Symbolics.variables(:dG, 1:nx, 1:np_considered)
    _G = Symbolics.variables(:G, 1:nx, 1:np_considered)

    dfdx  = Symbolics.jacobian(_dynamics, _x)
    dfddx = Symbolics.jacobian(_dynamics, _dx)
    dfdp  = Symbolics.jacobian(_dynamics, _p)

    dGdt = is_dae ?  dfdp .+ dfdx * _G  .+ dfddx * _dG : dfdp + dfdx * _G

    _obs = observed(_x, _p, _t)
    _w = Symbolics.variables(:w, 1:length(_obs))
    _dhxG = Symbolics.variables(:dhxG, 1:length(_obs), 1:np_considered, 1:np_considered)
    _hxG = Symbolics.variables(:hxG, 1:length(_obs), 1:np_considered, 1:np_considered)
    dobsdx = Symbolics.jacobian(_obs, _x)
    upper_triangle = triu(trues(np_considered, np_considered))
    dhxGdt = map(1:length(_obs)) do i
         if is_dae
            _dhxG[i,:,:][upper_triangle] .- ((dobsdx[i:i,:] * _G)' * (dobsdx[i:i,:] * _G))[upper_triangle]
         else
            ((dobsdx[i:i,:] * _G)' *(dobsdx[i:i,:] * _G))[upper_triangle]
         end
    end

    differential_variables_ = vcat(_dx, _dG[:], reduce(vcat, [_dhxG[i,:,:][upper_triangle][:] for i=1:length(_obs)]))
    variables_ = vcat(_x, _G[:], reduce(vcat, [_hxG[i,:,:][upper_triangle][:] for i=1:length(_obs)]))
    parameters_ = vcat(p_vector,_w)
    expressions_ = vcat(_dynamics, dGdt[:], reduce(vcat, [dhxGdt[i][:] for i=1:length(_obs)]))

    iip_idx = iip ? 2 : 1
    aug_fun = begin
        if !is_dae
            Symbolics.build_function(expressions_, variables_, parameters_, _t; expression=Val{false}, cse=true)[iip_idx]
        else
            Symbolics.build_function(expressions_, differential_variables_, variables_, parameters_, _t; expression=Val{false}, cse=true)[iip_idx]
        end
    end
    aug_u0 = vcat(u0, zeros(length(variables_) - length(u0)))
    aug_p = vcat(prob.p, ones(length(_w)))

    scache = SymbolCache(variables_, vcat(full_p, _w), [_t])
    dfun = is_dae ? DAEFunction(aug_fun, sys=scache) : ODEFunction(aug_fun, sys=scache)
    !is_dae && return ODEProblem{iip}(dfun, aug_u0, tspan, aug_p; prob.kwargs...)

    aug_du0 = vcat(prob.du0, zeros(length(differential_variables_) - length(prob.du0)))
    aug_diff_vars = isnothing(prob.differential_vars) ? nothing : vcat(prob.differential_vars, trues(length(differential_variables_) - length(prob.du0)))
    return DAEProblem{iip}(dfun, aug_du0, aug_u0, tspan, aug_p; differential_vars = aug_diff_vars, prob.kwargs...)
end

"""
$(METHODLIST)

Augments dynamics with the differential equations for sensitivities of parameters `params`.
This method is used in cases when measurements are taken at discrete time points.
Returns either an ODEProblem or a DAEProblem, depending on the type of the original problem.
"""
function augment_dynamics_only_sensitivities(prob::SciMLBase.AbstractDEProblem;
            tspan=prob.tspan, control_indices = Int64[],
            params = setdiff(eachindex(prob.p), control_indices), observed = (u,p,t) -> u[eachindex(prob.u0)])

   is_dae = isa(prob, SciMLBase.DAEProblem)
    u0 = prob.u0
    nx, np, nc, np_considered = length(prob.u0), length(prob.p), length(control_indices), length(params)

    iip = SciMLBase.isinplace(prob)
    _dx = Symbolics.variables(:dx, 1:nx)
    _x = Symbolics.variables(:x, 1:nx)
    _p = Symbolics.variables(:p, 1:np)
    _t = Symbolics.variable(:t)

    _dynamics = begin
        if iip
            if !is_dae
                prob.f.f(_dx, _x, _p, _t)
                _dx
            else
                out = Symbolics.variables(:out, 1:nx)
                prob.f.f(out, _dx, _x, _p, _t)
                out
            end
        else
            if !is_dae
                prob.f.f(_x, _p, _t)
            else
                prob.f.f(_dx, _x, _p, _t)
            end
        end
    end

    _dG = Symbolics.variables(:G, 1:nx, 1:np_considered)
    _G = Symbolics.variables(:G, 1:nx, 1:np_considered)

    dfdx = Symbolics.jacobian(_dynamics, _x)
    dfddx = Symbolics.jacobian(_dynamics, _dx)
    dfdp = Symbolics.jacobian(_dynamics, _p[params])

    dGdt = is_dae ?  dfdp + dfdx * _G  + dfddx * _dG : dfdp + dfdx * _G
    _obs = observed(_x, _p, _t)
    _w = Symbolics.variables(:w, 1:length(_obs))

    differential_variables_ = vcat(_dx, _dG[:])
    variables_ = vcat(_x, _G[:])
    parameters_ = vcat(_p,_w)
    expressions_ = vcat(_dynamics, dGdt[:])

    iip_idx = iip ? 2 : 1
    aug_fun = begin
        if !is_dae
            Symbolics.build_function(expressions_, variables_, parameters_, _t; expression=Val{false}, cse=true)[iip_idx]
        else
            Symbolics.build_function(expressions_, differential_variables_, variables_, parameters_, _t; expression=Val{false}, cse=true)[iip_idx]
        end
    end
    #dfun = eval(aug_fun)
    aug_u0 = vcat(u0, zeros(length(variables_) - length(u0)))
    aug_p = vcat(prob.p, ones(length(_w)))

    scache = SymbolCache(variables_, parameters_, [_t])
    dfun = is_dae ? DAEFunction(aug_fun, sys=scache) : ODEFunction(aug_fun, sys=scache)

    !is_dae && return ODEProblem{iip}(dfun, aug_u0, tspan, aug_p; prob.kwargs...)

    aug_du0 = vcat(prob.du0, zeros(length(differential_variables_) - length(prob.du0)))
    aug_diff_vars = isnothing(prob.differential_vars) ? nothing : vcat(prob.differential_vars, trues(length(differential_variables_) - length(prob.du0)))
    return DAEProblem{iip}(dfun, aug_du0, aug_u0, tspan, aug_p; differential_vars = aug_diff_vars, prob.kwargs...)
end

function augment_dynamics_for_oed(layer::Union{SingleShootingLayer,MultipleShootingLayer};
                params = get_params(layer),
                observed::Function = (u,p,t) -> u[eachindex(get_problem(layer).u0)],
                measurement_points=nothing)

    prob = get_problem(layer)
    tspan = get_tspan(layer)
    _, control_indices = get_controls(layer)
    fixed = is_fixed(layer)
    discrete = !isnothing(measurement_points)
    discrete && return augment_dynamics_only_sensitivities(prob, tspan=tspan, control_indices=control_indices, params=params, observed=observed)
    fixed && return augment_dynamics_unweighted_fisher(prob, tspan=tspan, control_indices=control_indices, params=params, observed=observed)
    return augment_dynamics_full(prob, tspan=tspan, control_indices=control_indices, params=params, observed=observed)
end

"""
$(METHODLIST)
Augments dynamics with the differential equations for sensitivities and Fisher information
matrix of parameters `params`. Returns either a SingleShootingLayer or a MultipleShootingLayer,
depending on the type of the layer passed to this function.
"""
function augment_layer_for_oed(layer::Union{SingleShootingLayer, MultipleShootingLayer};
        params = get_params(layer),
        observed::Function=(u,p,t) -> u[eachindex(get_problem(layer).u0)],
        dt = isnothing(layer.controls) ? (-)(reverse(get_tspan(layer))...)/100 : first(diff(first(get_controls(layer)[1]).t)),
        measurement_points = nothing)

    prob_original = get_problem(layer)
    nh = length(observed(prob_original.u0, prob_original.p, prob_original.tspan[1]))
    prob = augment_dynamics_for_oed(layer; params = params, observed=observed, measurement_points=measurement_points)
    controls, control_idxs = get_controls(layer)
    tspan = get_tspan(layer)
    np = length(prob.p)
    control_indices = vcat(control_idxs, [i for i=1:np if i > np - nh])

    tunable_ic = get_tunable(layer)
    timegrid = isnothing(measurement_points) ? collect(first(tspan):dt:last(tspan))[1:end-1] : measurement_points
    sampling_controls = map(Tuple(1:nh)) do i
        ControlParameter(timegrid, name=Symbol("w_$i"), controls=ones(length(timegrid)), bounds=(0.,1.))
    end

    unrestricted_controls_old = begin
        if isa(layer, SingleShootingLayer)
            isnothing(controls) ? [] : controls
        else
            if !isnothing(controls)
                merge_ms_controls(layer)
            end
        end
    end

    new_controls = (unrestricted_controls_old..., sampling_controls...,)
    if typeof(layer) <: SingleShootingLayer
        return SingleShootingLayer(prob, layer.algorithm, control_indices, new_controls;
                    tunable_ic = tunable_ic, bounds_ic = layer.bounds_ic)
    else
        intervals = layer.shooting_intervals
        points = vcat(first.(intervals), last.(intervals)) |> sort! |> unique!
        nodes_lb, nodes_ub = layer.bounds_nodes
        aug_dim = length(setdiff(eachindex(prob.u0), eachindex(get_problem(layer).u0)))
        aug_nodes_lb, aug_nodes_ub = vcat(nodes_lb, -Inf*ones(aug_dim)), vcat(nodes_ub, Inf*ones(aug_dim))
        return MultipleShootingLayer(prob, first(layer.layers).algorithm, control_indices, new_controls, points;
                    tunable_ic = tunable_ic, bounds_ic = first(layer.layers).bounds_ic, bounds_nodes = (aug_nodes_lb,aug_nodes_ub))
    end
end

"""
    symmetric_from_vector(x)
Computes the symmetric matrix encoded in the vector `x`. It is assumed that `x` contains
the upper triangle matrix.
"""
function symmetric_from_vector(x::AbstractArray)
    n = Int(sqrt(2 * size(x, 1) + 0.25) - 0.5)
    Symmetric([x[i <= j ? Int(j * (j - 1) / 2 + i) : Int(i * (i - 1) / 2 + j)]  for i in 1:n, j in 1:n])
end

"""
$(METHODLIST)
Sort keys of the symbol cache by their indices. The `identifier` is used to filter the correct
subset of keys, e.g., the Fisher information matrix variables by "F", or sensitivities by "G".
Matrix-valued variables, such as the sensitivities can be reshaped via kwarg `_reshape`.
"""
function sort_variables(keys; identifier="F", _reshape=false)
    reverse_index_map = Dict(value => value == '₋' ? key : parse(Int, string(key)) for (key, value) in Symbolics.IndexMap)
    indices_F = last.(split.(string.(keys), identifier))
    length(keys) == 1 && begin
        return _reshape ? reshape(keys, (1,1)) : keys
    end
    indices_rows, indices_columns = first.(split.(indices_F, "ˏ")), last.(split.(indices_F, "ˏ"))
    indices_integer = map(eachindex(indices_rows)) do i
        idxs_rows = map(collect(indices_rows[i])) do row_char
            reverse_index_map[row_char]
        end
        idx_cols = map(collect(indices_columns[i])) do col_char
            reverse_index_map[col_char]
        end
        sum(idxs_rows .* [10^(length(idxs_rows)-i) for i=1:length(idxs_rows)]), sum(idx_cols .* [10^(length(idx_cols)-i) for i=1:length(idx_cols)])
    end

    I, J = reduce(vcat, first.(indices_integer)), reduce(vcat, last.(indices_integer))
    sortbyJ = sortperm(J)
    keys_sorted_by_J = keys[sortbyJ]
    J = J[sortbyJ]
    I = I[sortbyJ]
    sorted_vars = reduce(vcat, map(sort(unique(J))) do col
        idxs = J .== col
        sortbyI_for_given_col = sortperm(I[idxs])
        keys_sorted_by_J[idxs][sortbyI_for_given_col]
    end)

    if _reshape
        nx, np = maximum(I), maximum(J)
        return reduce(hcat, [sorted_vars[(i-1)*nx+1:i*nx] for i=1:np])
    else
         return sorted_vars
    end
end

"""
    fisher_variables(layer)
Returns the symbols of all Fisher information matrix variables in `layer`. Used for indexing
of solutions.
"""
function fisher_variables(layer::Union{SingleShootingLayer, MultipleShootingLayer})
    st = LuxCore.initialstates(Random.default_rng(), layer)
    st = isa(layer, SingleShootingLayer) ? st : st.layer_1
    keys_ = collect(keys(st.symcache.variables))
    idx = findall(Base.Fix2(startswith, "F"), collect(string.(keys_)))
    fsym = keys_[idx]
    isempty(fsym) && return nothing
    return sort_variables(fsym; identifier="F")
end

"""
    sensitivity_variables(layer)
Returns the symbols of all sensitivity variables in `layer`. Used for indexing
of solutions.
"""
function sensitivity_variables(layer::Union{SingleShootingLayer, MultipleShootingLayer})
    st = LuxCore.initialstates(Random.default_rng(), layer)
    st = isa(layer, SingleShootingLayer) ? st : st.layer_1
    keys_ = collect(keys(st.symcache.variables))
    idx = findall(Base.Fix2(startswith, "G"), collect(string.(keys_)))
    Gsym = keys_[idx]
    isempty(Gsym) && return nothing
    sort_variables(Gsym; identifier="G", _reshape=true)
end

"""
    observed_sensitivity_product_variables(layer, observed_idx)
Returns the symbols of all unweighted Fisher information matrix variables of observation
function `observed_idx` in `layer`.
Used for indexing of solutions in cases, when `layer` is fixed.
"""
function observed_sensitivity_product_variables(layer::SingleShootingLayer, observed_idx::Int)
    string_idx = string(observed_idx)
    char_idx = map(x -> only(x), string_idx)
    subscripts_idx = map(x -> Symbolics.IndexMap[x], char_idx)
    st = LuxCore.initialstates(Random.default_rng(), layer)
    st = isa(layer, SingleShootingLayer) ? st : st.layer_1
    keys_ = collect(keys(st.symcache.variables))
    idx = findall(Base.Fix2(startswith, string(Symbol("hxG", subscripts_idx))), collect(string.(keys_)))
    return sort_variables(keys_[idx]; identifier=string(Symbol("hxG", subscripts_idx, "ˏ")))
end