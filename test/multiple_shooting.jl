using Corleone
using OrdinaryDiffEqTsit5
using Test
using Random
using LuxCore
using ComponentArrays

function lotka_dynamics(u, p, t)
    return [u[1] - p[2] * prod(u[1:2]) - 0.4 * p[1] * u[1];
            -u[2] + p[3] * prod(u[1:2]) - 0.2 * p[1] * u[2];
            (u[1]-1.0)^2 + (u[2] - 1.0)^2]
end

tspan = (0., 12.)
u0 = [0.5, 0.7, 0.]
p0 = [0.0, 1.0, 1.0]

prob = ODEProblem(lotka_dynamics, u0, tspan, p0)

cgrid = collect(0.0:0.1:11.9)
N = length(cgrid)
control = ControlParameter(
    cgrid, name = :fishing, bounds=(0.0,1.0), controls = zeros(N)
)

# Multiple Shooting
shooting_points = [0.0, 3.0, 6.0, 9.0]
layer = MultipleShootingLayer(prob, Tsit5(), [1], (control,), shooting_points)

Ni = N / length(shooting_points) |> Int
np_without_controls = length(setdiff(eachindex(prob.p), layer.layers[1].control_indices))
nx = length(prob.u0)
ps, st = LuxCore.setup(Random.default_rng(), layer)
p = ComponentArray(ps)
lb, ub = Corleone.get_bounds(layer)

@testset "General Multiple shooting tests" begin
    @test Corleone.is_fixed(layer) == false
    @test length(ps) == 4 # shooting stages
    @test isempty(ps.layer_1.u0) # initial condition is not tunable
    @test all([length(getproperty(p, Symbol("layer_$i")).u0) == 3 for i=2:4]) # ICs of subsequent layers are tunable

    blocks = cumsum(vcat(0, Ni+np_without_controls, [Ni+np_without_controls+nx for _ = 2:4]))
    @test Corleone.get_block_structure(layer) == blocks
end