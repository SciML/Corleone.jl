using CorleoneCore
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using OrdinaryDiffEq
using Test

@variables x(..) = 0.5 [bounds=(0.0,5.0)]
@variables c(..)=0.5 [input = true, bounds=(-1,1)]
@parameters p = 1.0 [tunable = false]
tspan = (0.0,1.0)
@named first_order = ODESystem(
    [
        D(x(t)) ~ - p * x(t) + c(t),
    ], t, [x(t), c(t)], [p];
    costs=Num[∫((x(t) - 1)^2)],
    consolidate=sum,
    tspan=tspan,
    constraints = []
)

N = 10
controlmethod = DirectControlCallback(
    c(t) => (; timepoints=collect(0.0:0.1:0.9), defaults=collect(LinRange(0,1.0,N)))
)

newsys, subs = CorleoneCore.extend_costs(first_order)
sys = tearing(controlmethod(newsys))

tpoints = [0.0, 0.5,1.0]
grid = ShootingGrid(tpoints, ConstantInitialization(sys, tpoints; init_values = [x(t)=> 1.0]))

ctrl_sys = complete(grid(sys))
pred = OCPredictor{false}(ctrl_sys, Tsit5())

@testset "Basic permutation tests" begin
    @test pred.permutation.blocks == [0, 6, 12, 13]
    @test pred.permutation.bounds_permuted[1] == (0.5, 0.5) # Fixed initial value for x
    @test pred.permutation.bounds_permuted[2] == (-1, 1.0)  # Then controls
    @test pred.permutation.bounds_permuted[end-1] == (-1, 1.0) # Final control
    @test pred.permutation.bounds_permuted[end] == (0.0, 5.0)  # Terminal shooting node
end
