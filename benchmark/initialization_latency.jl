# Run in a fresh process after precompiling Corleone. Package loading is excluded.
using Corleone, LuxCore, Random, SciMLBase

function initialization_workload()
    problem = ODEProblem((u, p, t) -> p .* u, [1.0, 2.0, 3.0], (0.0, 2.0), [0.2, 0.1, 0.0])
    rng = MersenneTwister(42)
    sys = ControlSymbolCache(problem)
    conditions = InitialConditions(
        problem,
        Tunable(1; kind = Val(:variable)) => [0.0, 1.0],
        Constant(2; kind = Val(:variable)) => [1.0]; sys
    )
    parameters = Parameter(problem, Tunable(1; kind = Val(:parameter)) => [0.5, 1.5]; fixed = [3], sys)
    ps_ic, st_ic = LuxCore.setup(rng, conditions)
    ps_p, st_p = LuxCore.setup(rng, parameters)
    for t in (0.0, 0.5, 1.0, 1.5)
        conditions((problem.u0, t), ps_ic, st_ic; matching = t != 0.0)
        parameters(t, ps_p, st_p)
    end
    return Corleone.get_timestops(conditions, ps_ic, st_ic, Float64),
        Corleone.get_timestops(parameters, ps_p, st_p, Float64)
end

first_call = @timed initialization_workload()
@assert first_call.value == ([0.0, 1.0], [0.5, 1.5])
warm_call = @timed initialization_workload()
println("first_seconds,first_bytes,warm_seconds,warm_bytes")
println(first_call.time, ",", first_call.bytes, ",", warm_call.time, ",", warm_call.bytes)
