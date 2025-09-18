struct OEDLayer{fixed,L,O,D} <: LuxCore.AbstractLuxLayer
    layer::L
    observed::O
    dimensions::D
end


function OEDLayer(layer::Union{SingleShootingLayer,MultipleShootingLayer};
                    observed = (u,p,t) -> u,
                    params = get_params(layer),
                    dt = (-)(reverse(tspan)...)/100
                    )

    prob = get_problem(layer)
    nx, np, nc, np_considered = length(prob.u0), length(prob.p), length(layer.control_indices), length(params)

    fixed = isempty(layer.control_indices) && isempty(layer.tunable_ic)

    oed_layer = augment_layer_for_oed(layer, params=params, observed=observed, dt=dt)

    obs = begin
        if fixed
            x, p, t = Symbolics.variables(:x, 1:nx), Symbolics.variables(:p, 1:np), Symbolics.variable(:t)

            h = observed(x,p,t)
            hx = Symbolics.jacobian(h, x)
            hx_fun = Symbolics.build_function(hx, x, p, t, expression = Val{false})[1]

            (h = observed, hx = hx_fun)
        else
            nothing
        end
    end

    dimensions = (np = np, nh = length(observed(prob.u0, prob.p, first(prob.tspan))),
                  np_fisher = np_considered, nc = nc, nx = nx)
    return OEDLayer{fixed, typeof(oed_layer), typeof(obs), typeof(dimensions)}(oed_layer, obs, dimensions)
end

LuxCore.initialparameters(rng::Random.AbstractRNG, layer::OEDLayer) = LuxCore.initialparameters(rng, layer.layer)
LuxCore.initialstates(rng::Random.AbstractRNG, layer::OEDLayer) = LuxCore.initialstates(rng, layer.layer)

function (layer::OEDLayer)(::Any, ps, st)
    layer.layer(nothing, ps, st)
end

(crit::AbstractCriterion)(oedlayer::OEDLayer{true, <:SingleShootingLayer, <:Any, <:Any}) = begin
    ps, st = LuxCore.setup(Random.default_rng(), oedlayer)
    sols, _ = oedlayer(nothing, ps, st)
    sens = reshape(sort(CorleoneCore.sensitivity_variables(oedlayer.layer), by= x -> split(string(x), "ˏ")[2]), (oedlayer.dimensions.nx,oedlayer.dimensions.np_fisher))
    nc = vcat(0, cumsum(map(x -> length(x.t), oedlayer.layer.controls))...)
    (p, ::Any) -> let sols = sols, prob_p = oedlayer.layer.problem.p, sampling = oedlayer.layer.controls, dims = oedlayer.dimensions, hx = oedlayer.observed.hx, ax = getaxes(ComponentArray(ps)), nc=nc
        ps = ComponentArray(p, ax)
        F = Symmetric(sum(map(enumerate(sampling)) do (widx, wi)
            sum(map(enumerate(sols.t[1:end-1])) do (i, ti)
                cidx = findlast(t -> ti >= t, wi.t)
                hxG = hx(sols[i][1:dims.nx], prob_p, ti)[widx:widx,:] * sols[sens][i]
                (sols.t[i+1] - ti) * ps.controls[nc[widx]+1:nc[widx+1]][cidx] * hxG' * hxG
            end)
        end))
        crit(F)
    end
end

(crit::AbstractCriterion)(oedlayer::OEDLayer{false}) = begin

end
