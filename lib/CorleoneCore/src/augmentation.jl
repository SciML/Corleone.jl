get_problem(layer::SingleShootingLayer) = layer.problem
get_problem(layer::MultipleShootingLayer) = get_problem(first(layer.layers))
get_controls(layer::SingleShootingLayer) = (layer.controls, layer.control_indices)
get_controls(layer::MultipleShootingLayer) = get_controls(first(layer.layers))
get_tspan(layer::SingleShootingLayer) = layer.problem.tspan
get_tspan(layer::MultipleShootingLayer) = (first(first(layer.layers).problem.tspan), last(last(layer.layers).problem.tspan))

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

    @info _dx dfdx dfdp
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