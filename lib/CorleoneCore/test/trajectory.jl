using CorleoneCore
using OrdinaryDiffEq
using Test

function dynamics(u, p, t)
    -p * u[1]
end
u0 = [-1.0]
tspan = (0.0, 1.0)
p = [-0.1]
prob = ODEProblem(dynamics, u0, tspan, p)

@testset "Basic trajectory tests" begin
    sol = solve(prob, Tsit5(), saveat=0.1)
    @test_nowarn @inferred Trajectory(sol)
    traj = Trajectory(sol)
    @test eltype(traj) == Float64
    @test Array(sol) == traj.states
    @test sol.t == traj.time
    @test prob.p == traj.parameters
    @test all(traj.retcodes)
    @test isnothing(traj.shooting_variables)
    @test !any(traj.special_variables.pseudo_mayer)
    @test isempty(traj.mayer_variables)
    # Merge 
    sol1 = solve(prob, Tsit5(), tspan=(0.0, 0.5), saveat=0.1)
    sol2 = solve(prob, Tsit5(), u0=sol1[:, end], tspan=(0.5, 1.0), saveat=0.1)
    sols = map(Trajectory, (sol1, sol2))
    @test_nowarn @inferred merge(sols...)
    traj = merge(sols...)
    @test Array(sol) ≈ traj.states
    @test sol.t ≈ traj.time
    @test all(≈(sol1[:, end]), traj.shooting_variables...)
    @test !any(traj.special_variables.pseudo_mayer)
    @test isempty(traj.mayer_variables)
end





