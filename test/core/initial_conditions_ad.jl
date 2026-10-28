using Corleone, LuxCore, OrdinaryDiffEqTsit5, Random, Test, ForwardDiff, ReverseDiff, Zygote
using ComponentArrays, DiffEqBase
import SciMLBase

@testset "Initial-condition values and pass-through gradients" begin
    prob = ODEProblem((u, p, t) -> ones(2), [1.0, 2.0], (0.0, 4.0), [0.0])
    layer = InitialConditions(prob, Tunable(1) => [1.0, 3.0], Constant(2) => [2.0])
    ps, st = LuxCore.setup(MersenneTwister(1), layer)
    function objective(q, matching = true)
        params = (;
            initialconditions = (;
                u₁ = (; lookup = ps.initialconditions.u₁.lookup, parameter = (; ps = q[1:2])),
                u₂ = ps.initialconditions.u₂,
            ),
        )
        states = (;
            active = st.active, initialconditions = (;
                u₁ = st.initialconditions.u₁,
                u₂ = (;
                    states = st.initialconditions.u₂.states,
                    parameters = (; lookup = (;), parameter = (; ps = [q[5]])),
                ),
            ),
        )
        sum((0.0, 1.0, 2.0, 3.0)) do t
            (u, residuals), _ = layer((q[3:4], t), params, states; matching)
            sum(abs2, u) + sum(abs2, residuals)
        end
    end
    q = [10.0, 30.0, 4.0, 5.0, 20.0]
    # Every selected value contributes to its output and matching residual;
    # unselected incoming values remain active through the pass-through branch.
    expected = [32.0, 112.0, -48.0, 30.0, 40.0]
    @test ForwardDiff.gradient(objective, q) ≈ expected
    @test ReverseDiff.gradient(objective, q) ≈ expected
    @test only(Zygote.gradient(objective, q)) ≈ expected
    for gradient in (ForwardDiff.gradient, ReverseDiff.gradient, (f, q) -> only(Zygote.gradient(f, q)))
        @test gradient(q -> objective(q, false), q) ≈ [20.0, 60.0, 16.0, 30.0, 40.0]
    end
end

@testset "ForwardDiff with the first lifted value after the start" begin
    prob = ODEProblem((u, p, t) -> [p[1] + zero(u[1])], [1.0], (0.0, 3.0), [1.0])
    layer = ShootingLayer(
        prob, Tsit5(),
        Tunable(1; kind = Val(:variable)) => [1.0, 2.0]
    )
    ps, st = LuxCore.setup(MersenneTwister(1), layer)
    function endpoint(q)
        parameters = merge(
            ps, (;
                initialcondition = (;
                    initialconditions = (;
                        u₁ = (; lookup = (;), parameter = (; ps = q)),
                    ),
                ),
            )
        )
        trajectory, _ = layer(nothing, parameters, st)
        return last(trajectory.u)[1]
    end
    @test ForwardDiff.gradient(endpoint, [10.0, 20.0]) ≈ [0.0, 1.0]
end

# An unannotated scalar return checks inference through solve and trajectory assembly.
first_segment_objective(layer, ps, st) = last(first(layer(nothing, ps, st)).u)[1]

@testset "First segment infers with Dual overrides" begin
    prob = ODEProblem(
        (u, p, t) -> [p[1] * u[1], p[2]], [1.0, 0.0], (0.0, 3.0), [0.2, 0.5];
        saveat = [0.25, 1.25, 2.25], tstops = [0.5, 1.5, 2.5]
    )
    for ensemblealg in (EnsembleSerial(), EnsembleThreads())
        layer = ShootingLayer(
            prob, Tsit5(),
            Tunable(1; kind = Val(:variable)) => [1.0, 2.0],
            Tunable(2; kind = Val(:variable)) => [1.0, 2.0]; ensemblealg
        )
        ps, st = LuxCore.setup(MersenneTwister(1), layer)
        dual = ForwardDiff.Dual{Nothing}.(ps.parameter.parameters.p₁.parameter.ps, 1.0)
        parameters = merge(ps, (; parameter = (; parameters = merge(ps.parameter.parameters, (; p₁ = (; lookup = (;), parameter = (; ps = dual)))))))
        value = @inferred first_segment_objective(layer, parameters, st)
        @test value isa eltype(dual)
        @test isfinite(ForwardDiff.partials(value)[1])
        @test (@inferred first_segment_objective(layer, ps, st)) isa Float64
        trajectory, _ = layer(nothing, parameters, st)
        @test [0.25, 1.25, 2.25] ⊆ trajectory.t
    end
end

@testset "ReverseDiff scalar packing and full pass-through solves" begin
    function dynamics!(du, u, p, t)
        du[1] = p[1] * u[1]
        return nothing
    end
    problem = ODEProblem(
        ODEFunction{true, SciMLBase.FullSpecialize}(dynamics!), [1.0], (0.0, 2.0), [0.2];
        sensealg = SensitivityADPassThrough(), abstol = 1.0e-11, reltol = 1.0e-11
    )
    for lifted in (false, true)
        definitions = lifted ? (Tunable(1; kind = Val(:variable)) => [0.75, 1.5],) : ()
        layer = ShootingLayer(problem, Tsit5(), Tunable(1; kind = Val(:parameter)) => [1.0], definitions...)
        original_ps, original_st = LuxCore.setup(MersenneTwister(1), layer)
        ps, st = lifted ? add_parameter_segments!(layer, original_ps, original_st) : (original_ps, original_st)
        parameters = ComponentArray(ps)
        axes = getaxes(parameters)
        function loss(q)
            traj, _ = layer(nothing, ComponentArray(q, axes), st)
            return last(traj.u)[1] + 0.1 * sum(abs2, shooting_constraints(traj))
        end
        function check_packing(q)
            params = ComponentArray(q, axes)
            ((incoming, _), _) = layer.initialcondition((problem.u0, 0.0), params.initialcondition, st.initialcondition)
            p, _ = layer.parameter(0.0, params.parameter, st.parameter)
            @test incoming isa Vector
            @test p isa Vector{<:ReverseDiff.TrackedReal}
            if lifted
                ic = layer.initialcondition.initialconditions.u₁
                initial, _ = ic((0.0, 0.0), params.initialcondition.initialconditions.u₁, st.initialcondition.initialconditions.u₁)
                reset, _ = ic((initial, 0.75), params.initialcondition.initialconditions.u₁, st.initialcondition.initialconditions.u₁)
                @test typeof(initial) === typeof(reset)
            end
            return sum(incoming) + sum(p)
        end
        q = collect(parameters)
        ReverseDiff.gradient(check_packing, q)
        for input in (q, q .+ 0.01)
            original = copy(input)
            @test ReverseDiff.gradient(loss, input) ≈ ForwardDiff.gradient(loss, input) rtol = 1.0e-6 atol = 1.0e-7
            @test input == original
        end
    end
end
