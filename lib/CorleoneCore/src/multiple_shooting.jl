struct MultipleShootingLayer{L,SI,E} <: LuxCore.AbstractLuxLayer
    layers::L
    shooting_intervals::SI
    ensemble_alg::E
end

function MultipleShootingLayer(prob, alg, tunable, control_indices, controls, shooting_points; ensemble_alg = EnsembleSerial())
    tspan = prob.tspan
    shooting_points = vcat(tspan..., shooting_points) |> unique! |> sort!
    shooting_intervals = [(t0,t1) for (t0,t1) in zip(shooting_points[1:end-1], shooting_points[2:end])]

    _tunable = vcat(tunable, [collect(1:length(prob.u0)) for _ in 1:length(shooting_intervals)])
    layers = [SingleShootingLayer(remake(prob, tspan = tspani), alg, _tunable[i], control_indices, restrict_controls(controls, tspani...)) for (i, tspani) in enumerate(shooting_intervals)]

    MultipleShootingLayer{typeof(layers), typeof(shooting_intervals), typeof(ensemble_alg)}(layers, shooting_intervals, ensemble_alg)
end


function LuxCore.initialparameters(rng::Random.AbstractRNG, mslayer::MultipleShootingLayer)
    layer_names = Tuple([Symbol("layer_$i") for i=1:length(mslayer.layers)])
    layer_ps    = Tuple([LuxCore.initialparameters(rng, layer) for layer in mslayer.layers])
    NamedTuple{layer_names}(layer_ps)
end

function LuxCore.initialstates(rng::Random.AbstractRNG, mslayer::MultipleShootingLayer)
    layer_names = Tuple([Symbol("layer_$i") for i=1:length(mslayer.layers)])
    layer_st    = Tuple([LuxCore.initialstates(rng, layer) for layer in mslayer.layers])
    NamedTuple{layer_names}(layer_st)
end

function (layer::MultipleShootingLayer)(::Any, ps, st)
    prob = SingleShootingProblem(first(layer.layers), ps.layer_1, st.layer_1)
    remaker = let ps = ps, st=st, names = keys(ps)
        function (prob, i, repeat)
            current = names[i]
            p_current = getproperty(ps, current)
            st_current = getproperty(st, current)
            prob_current = remake(prob; ps=p_current, st=st_current)
            prob_current
        end
    end
    return solve(EnsembleProblem(prob, prob_func=remaker, output_func = (sol, i) -> (sol[1], false)),
            DummySolve(),layer.ensemble_alg; trajectories = length(layer.layers)), st
end



# MultipleShootingLayer ist Container für SingleShootingLayer, st müssen gefixt werden, ps nicht
# Machen so wie Wrapper Struct, Common Solve für ShootingProblem
# init() definieren init(layer,alg,ps,st) -> ShootingProblem() -> solve!(ShootingProblem...)::NTuple
# Oder pmap? Gucken bei Lux?