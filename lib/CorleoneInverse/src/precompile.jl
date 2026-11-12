@setup_workload begin
    problem = SciMLBase.ODEProblem((u, p, t) -> [p[1]], [0.0], (0.0, 1.0), [0.5])
    shooting = ShootingLayer(problem, Tsit5())
    data = Experiment(1 => ([0.0, 0.5, 1.0], [0.0, 0.5, 1.0]))
    experiments = Experiments(problem; experiments = (run = data,))
    ad = AutoForwardDiff()
    @compile_workload begin
        model = CorleoneDynamicOptProblem(shooting, experiments; rng = Random.MersenneTwister(42))
        least_squares = SciMLBase.NonlinearLeastSquaresProblem(model)
        SciMLBase.solve(least_squares, LevenbergMarquardt(; autodiff = ad))

        optimization = SciMLBase.OptimizationProblem(model, ad; ps = ComponentArray(model.ps))
        unconstrained = SciMLBase.remake(optimization; lb = nothing, ub = nothing)
        SciMLBase.solve(unconstrained, Adam(0.01); maxiters = 1)
    end
end
