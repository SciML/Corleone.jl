using Test, Corleone, ComponentArrays, OrdinaryDiffEqTsit5, SparseArrays
import OptimizationBase

@testset "OptimizationBase canonical Lagrangian coordinates" begin
    problem = ODEProblem((u, p, t) -> p, [1.0], (0.0, 1.0), [0.2])
    layer = ShootingLayer(problem, Tsit5(), Tunable(1; default = 0.2) => [0.25, 0.75])
    model = CorleoneDynamicOptProblem(layer, :((p₁(0.0) + p₁(1.0))^2))
    prototype = lagrangian_hessian_sparsity(model)
    rows, cols = OptimizationBase.lag_hess_structure(model)
    @test (rows, cols) == OptimizationBase.lag_hess_structure(prototype)
    @test collect(zip(rows, cols)) == [(1, 1), (1, 3), (3, 3)]
    @test all(rows .<= cols)
    @test issorted(collect(zip(cols, rows))) # CSC column order, not row order.
    parameters = ComponentArray(model.ps)
    @test OptimizationBase.lag_hess_structure(model; ps = parameters) == (rows, cols)
end
