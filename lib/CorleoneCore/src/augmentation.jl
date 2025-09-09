function augment_dynamics_for_oed(layer::Union{SingleShootingLayer,MultipleShootingLayer};
                params = get_params(layer),
                observed::Function = (u,p,t) -> u)

    prob = get_problem(layer)
    u0, tspan = prob.u0, get_tspan(layer)
    controls, control_indices = get_controls(layer)
    nx, np, nc, np_considered = length(prob.u0), length(prob.p), length(control_indices), length(params)

    iip = SciMLBase.isinplace(prob)
    _x = Symbolics.variables(:x, 1:nx)
    _p = Symbolics.variables(:p, 1:np)
    _t = Symbolics.variable(:t)

    _dx = begin
        if iip
            __dx = Symbolics.variables(:dx, 1:nx)
            prob.f.f(__dx, _x, _p, _t)
            __dx
        else
            prob.f.f(_x, _p, _t)
        end
    end


    _G = Symbolics.variables(:G, 1:nx, 1:np_considered)
    _F = Symbolics.variables(:F, 1:np_considered, 1:np_considered)
    dfdx = Symbolics.jacobian(_dx, _x)
    dfdp = Symbolics.jacobian(_dx, _p[params])

    dGdt = dfdp + dfdx * _G
    _obs = observed(_x, _p, _t)
    _w = Symbolics.variables(:w, 1:length(_obs))
    dobsdx = Symbolics.jacobian(_obs, _x)

    upper_triangle = triu(trues(np_considered, np_considered))
    dFdt = sum(map(1:length(_obs)) do i
        _w[i] * (dobsdx[i:i,:] * _G)' * (dobsdx[i:i,:] * _G)
    end)[upper_triangle]

    iip_idx = iip ? 2 : 1
    aug_fun = Symbolics.build_function(vcat(_dx, dGdt[:], dFdt), vcat(_x, _G[:], _F[upper_triangle]), vcat(_p,_w), _t)[iip_idx]
    dfun = eval(aug_fun)
    aug_u0 = vcat(u0, zeros(nx*np_considered+Int((np_considered*(np_considered+1)/2))))
    aug_p = vcat(prob.p, ones(length(_w)))

    scache = SymbolCache(vcat(_x, _G[:], _F[upper_triangle]), vcat(_p,_w), [_t])
    dfun = ODEFunction(dfun, sys=scache)
    ODEProblem{iip}(dfun, aug_u0, tspan, aug_p)
end


function augment_layer_for_oed(layer::Union{SingleShootingLayer, MultipleShootingLayer};
        params = get_params(layer),
        observed::Function=(u,p,t) -> u,
        dt = first(diff(first(get_controls(layer)[1]).t)))


    prob = augment_dynamics_for_oed(layer; params = params, observed=observed)
    controls, control_idxs = get_controls(layer)
    tspan = get_tspan(layer)
    nh = length(observed(prob.u0, prob.p, prob.tspan[1]))
    np = length(prob.p)
    control_indices = vcat(control_idxs, [i for i=1:np if i > np - nh])

    tunable_ic = get_tunable(layer)
    timegrid = collect(first(tspan):dt:last(tspan))[1:end-1]
    sampling_controls = map(Tuple(1:nh)) do i
        ControlParameter(timegrid,name=Symbol("w_$i"), controls=ones(length(timegrid)))
    end

    unrestricted_controls_old = map(controls) do ci
        ControlParameter(timegrid, name=ci.name)
    end
    new_controls = (unrestricted_controls_old..., sampling_controls...,)
    if typeof(layer) <: SingleShootingLayer
        return SingleShootingLayer(prob, layer.algorithm, tunable_ic, control_indices, new_controls)
    else
        intervals = layer.shooting_intervals
        points = vcat(first.(intervals), last.(intervals)) |> sort! |> unique!
        return MultipleShootingLayer(prob, first(layer.layers).algorithm, tunable_ic, control_indices, new_controls, points)
    end

end

function symmetric_from_vector(x::AbstractArray)
    n = Int(sqrt(2 * size(x, 1) + 0.25) - 0.5)
    [x[i <= j ? Int(j * (j - 1) / 2 + i) : Int(i * (i - 1) / 2 + j)]  for i in 1:n, j in 1:n]
end

function fisher_variables(layer::Union{SingleShootingLayer, MultipleShootingLayer})
    st = LuxCore.initialstates(Random.default_rng(), layer)
    st = isa(layer, SingleShootingLayer) ? st : st.layer_1
    keys_ = collect(keys(st.symcache.variables))
    idx = findall(Base.Fix2(startswith, "F"), collect(string.(keys_)))
    sort(keys_[idx], by=string) ## TODO: TEST IF THIS RETURNS THE RIGHT ORDER WHEN APPLYING SYMMETRIC_FROM_VECTOR
end

function sensitivity_variables(layer::Union{SingleShootingLayer, MultipleShootingLayer})
    st = LuxCore.initialstates(Random.default_rng(), layer)
    st = isa(layer, SingleShootingLayer) ? st : st.layer_1
    keys_ = collect(keys(st.symcache.variables))
    idx = findall(Base.Fix2(startswith, "G"), collect(string.(keys_)))
    sort(keys_[idx], by=string) ## TODO: TEST IF THIS RETURNS THE RIGHT ORDER WHEN APPLYING SYMMETRIC_FROM_VECTOR
end