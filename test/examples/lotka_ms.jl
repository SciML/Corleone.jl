using Corleone, OrdinaryDiffEqTsit5, LuxCore, ComponentArrays, Random, Test
using Optimization, OptimizationMOI, Ipopt, ForwardDiff

@testset "Lotka lifted states and explicit continuity constraints" begin
    function dynamics!(du, u, p, t)
        du[1] = u[1] - p[2] * u[1] * u[2] - 0.4 * p[1] * u[1]
        du[2] = -u[2] + p[3] * u[1] * u[2] - 0.2 * p[1] * u[2]
        du[3] = (u[1] - 1)^2 + (u[2] - 1)^2
        return nothing
    end
    prob = ODEProblem(
        dynamics!, [0.5, 0.7, 0.0], (0.0, 12.0), [0.0, 1.0, 1.0];
        abstol = 1.0e-9, reltol = 1.0e-9
    )
    controls = Tunable(1; kind = Val(:parameter)) => collect(0.1:0.1:11.9)
    fixed = (
        Constant(2; kind = Val(:parameter)) => nothing,
        Constant(3; kind = Val(:parameter)) => nothing,
    )
    sequential = ShootingLayer(prob, Tsit5(), controls, fixed...)
    base_ps, base_st = LuxCore.setup(MersenneTwister(42), sequential)
    baseline, _ = sequential(nothing, base_ps, base_st)
    nodes = [3.0, 6.0, 9.0]
    # Lift either all states, or just populations and carry the cost continuously.
    for lifted in (1:3, 1:2)
        definitions = (Tunable(i; kind = Val(:variable)) => nodes for i in lifted)
        layer = ShootingLayer(
            prob, Tsit5(), controls, fixed..., definitions...;
            ensemblealg = EnsembleThreads()
        )
        ps, st = LuxCore.setup(MersenneTwister(42), layer)
        for (i, ic) in enumerate(values(ps.initialcondition.initialconditions))
            i in lifted || continue
            ic.parameter.ps .= [baseline.u[findfirst(==(t), baseline.t)][i] for t in nodes]
        end
        initial = ComponentArray(ps)
        axes = getaxes(initial)
        trajectory, _ = layer(nothing, ps, st)
        @test length(trajectory.reset_indices) == length(nodes)
        @test last(trajectory.u)[3] ≈ last(baseline.u)[3]
        @test reduce(vcat, trajectory.residuals) ≈ zeros(length(lifted) * length(nodes)) atol = 1.0e-10
        @test LuxCore.parameterlength(layer) == 120 + length(lifted) * length(nodes)

        objective(q, state) = last(first(layer(nothing, ComponentArray(q, axes), state)).u)[3]
        function constraints!(res, q, state)
            trajectory, _ = layer(nothing, ComponentArray(q, axes), state)
            res .= reduce(vcat, trajectory.residuals)
            return nothing
        end
        upper = similar(initial)
        fill!(upper, Inf)
        upper.parameter.parameters.p₁.parameter.ps .= 1.0
        nconstraints = length(lifted) * length(nodes)
        optfun = OptimizationFunction(objective, AutoForwardDiff(); cons = constraints!)
        optprob = OptimizationProblem(
            optfun, initial, st; lb = zero(initial), ub = upper,
            lcons = zeros(nconstraints), ucons = zeros(nconstraints)
        )
        result = solve(
            optprob, Ipopt.Optimizer(); max_iter = 300, tol = 1.0e-6,
            hessian_approximation = "limited-memory", print_level = 0
        )
        @test SciMLBase.successful_retcode(result)
        @test result.objective ≈ 1.344336 atol = 2.0e-4
        residuals = zeros(nconstraints)
        constraints!(residuals, result.u, st)
        @test maximum(abs, residuals) <= 1.0e-5
    end
end
