# Exercise the solver-independent numerical layers without adding a solver
# dependency. Solver-specific compilation belongs to the downstream application.
@setup_workload begin
    problem = ODEProblem((u, p, t) -> p .* u, [1.0, 2.0, 3.0], (0.0, 2.0), [0.2, 0.1, 0.0])
    rng = MersenneTwister(42)
    @compile_workload begin
        sys = ControlSymbolCache(problem)
        conditions = InitialConditions(
            problem,
            Tunable(1; kind = Val(:variable)) => [0.0, 1.0],
            Constant(2; kind = Val(:variable)) => [1.0]; sys
        )
        parameters = Parameter(
            problem, Tunable(1; kind = Val(:parameter)) => [0.5, 1.5];
            fixed = [3], sys
        )
        ps_ic, st_ic = LuxCore.setup(rng, conditions)
        ps_p, st_p = LuxCore.setup(rng, parameters)
        for t in (0.0, 0.5, 1.0, 1.5)
            conditions((problem.u0, t), ps_ic, st_ic; matching = t != 0.0)
            parameters(t, ps_p, st_p)
        end
        get_timestops(conditions, ps_ic, st_ic, Float64)
        get_timestops(parameters, ps_p, st_p, Float64)
    end
end
