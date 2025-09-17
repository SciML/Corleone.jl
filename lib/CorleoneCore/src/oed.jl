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