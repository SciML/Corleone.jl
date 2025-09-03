struct MultipleShootingLayer{L,SI} <: LuxCore.AbstractLuxLayer
    layers::L
    shooting_intervals::SI
end

function MultipleShootingLayer(prob, alg, tunable, control_indices, controls, shooting_points)


    tspan = prob.tspan
    shooting_points = vcat(tspan..., shooting_points) |> unique! |> sort!
    shooting_intervals = [(t0,t1) for (t0,t1) in zip(shooting_points[1:end-1], shooting_points[2:end])]

    _tunable = vcat(tunable, [collect(1:length(prob.u0)) for _ in 1:length(shooting_intervals)])
    @info _tunable
    @info shooting_intervals
    layers = [SingleShootingLayer(remake(prob, tspan = tspani), alg, _tunable[i], control_indices, controls) for (i, tspani) in enumerate(shooting_intervals)]


    MultipleShootingLayer{typeof(layers), typeof(shooting_intervals)}(layers, shooting_intervals)
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


# MultipleShootingLayer ist Container für SingleShootingLayer, st müssen gefixt werden, ps nicht
# Machen so wie Wrapper Struct, Common Solve für ShootingProblem
# init() definieren init(layer,alg,ps,st) -> ShootingProblem() -> solve!(ShootingProblem...)::NTuple
# Oder pmap? Gucken bei Lux?