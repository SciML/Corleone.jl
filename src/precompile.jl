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

        # Exercise the shared-grid layout used for multi-index annotations.
        # Defaults are vectors over the selected indices; each entry is itself
        # initialized over the common time grid.
        times = [0.5, 1.0]
        grouped_conditions = InitialConditions(
            problem,
            Tunable(
                [1, 3]; kind = Val(:variable), default = [[0.4, 0.6], [0.8, 1.0]]
            ) => times,
            Constant(2; kind = Val(:variable), default = [1.5, 1.6]) => times;
            sys
        )
        # Two parameter switches define three intervals.
        grouped_parameters = Parameter(
            problem,
            Tunable([1, 2]; default = [[0.2, 0.3, 0.4], [0.1, 0.2, 0.3]]) => times,
            Constant(3; default = [0.0, 0.0, 0.0]) => times;
            sys
        )
        ps_grouped_ic, st_grouped_ic = LuxCore.setup(rng, grouped_conditions)
        ps_grouped_p, st_grouped_p = LuxCore.setup(rng, grouped_parameters)
        for t in (0.0, 0.5, 1.0, 1.5)
            grouped_conditions((problem.u0, t), ps_grouped_ic, st_grouped_ic)
            grouped_parameters(t, ps_grouped_p, st_grouped_p)
        end
        get_timestops(grouped_conditions, ps_grouped_ic, st_grouped_ic, Float64)
        get_timestops(grouped_parameters, ps_grouped_p, st_grouped_p, Float64)
    end
end
