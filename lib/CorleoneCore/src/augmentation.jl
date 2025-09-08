function augment_dynamics_for_oed(layer::Union{SingleShootingLayer,MultipleShootingLayer};
                observed::Function = (u,p,t) -> u)

    prob = get_problem(layer)
    u0, tspan = prob.u0, get_tspan(layer)
    controls, control_indices = get_controls(layer)
    p_indices = [i for i in eachindex(prob.p) if i ∉ control_indices]
    nx, np, nc = length(prob.u0), length(prob.p), length(control_indices)

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


    _G = Symbolics.variables(:G, 1:nx, 1:np-nc)
    _F = Symbolics.variables(:F, 1:np-nc, 1:np-nc)


    dfdx = Symbolics.jacobian(_dx, _x)
    dfdp = Symbolics.jacobian(_dx, _p[p_indices])

    dGdt = dfdp + dfdx * _G
    _obs = observed(_x, _p, _t)
    _w = Symbolics.variables(:w, 1:length(_obs))
    dobsdx = Symbolics.jacobian(_obs, _x)

    dFdt = sum(map(1:length(_obs)) do i
        _w[i] * (dobsdx[i:i,:] * _G)' * (dobsdx[i:i,:] * _G)
    end)

    iip_idx = iip ? 2 : 1
    aug_fun = Symbolics.build_function(vcat(_dx, dGdt[:], dFdt[:]), vcat(_x, _G[:], _F[:]), vcat(_p,_w), _t)[iip_idx]
    dfun = eval(aug_fun)
    aug_u0 = vcat(u0, zeros(nx*(np-nc)+(np-nc)^2))
    aug_p = vcat(prob.p, ones(length(_w)))

    ODEProblem{iip}(dfun, aug_u0, tspan, aug_p)
end


function augment_layer_for_oed(layer::Union{SingleShootingLayer, MultipleShootingLayer};
        observed::Function=(u,p,t) -> u,
        dt = first(diff(first(get_controls(layer)[1]).t)))


    prob = augment_dynamics_for_oed(layer; observed=observed)
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
