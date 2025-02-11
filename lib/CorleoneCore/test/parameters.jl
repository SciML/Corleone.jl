using CorleoneCore
using LuxCore
using Random
using Test

rng = Random.seed!()
 
p20 = [3., 4., 5.]
tstops2 = [1., 3., 10.]
p10 = [1., 2.]

layers = (
    Parameter(p10), 
    Parameter(p20, tstops = tstops2),
)

container = ParameterContainer(layers...)

@test contains_timegrid_layer(container)
@test contains_tstop_layer(container)
@test !contains_saveat_layer(container)

ps, st = LuxCore.setup(rng, container)

teval = [0.8, 1.2, 2.5, 3.0, 8., 11.]
pidx = [1, 1, 1, 2, 2, 3]

for (ti, p2i) in zip(teval, pidx)
    p_t, st_ = container(ti, ps, st)
    @test st_ == st
    @test isa(p_t, NamedTuple)
    @test p_t.layer_1 == [1., 2.]
    @test p_t.layer_2 == selectdim(p20, 1, p2i)
end