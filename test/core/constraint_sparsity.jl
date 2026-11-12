using Test, Corleone, LuxCore, OrdinaryDiffEqTsit5, ComponentArrays, Random, ForwardDiff, SparseArrays
using RuntimeGeneratedFunctions

@testset "Structural constraint dependencies" begin
    problem = ODEProblem((u, p, t) -> [p[2] * u[2] + p[1], u[1] + p[1]], [1.0, 2.0], (0.0, 3.0), [0.2, 0.3]; abstol = 1.0e-10, reltol = 1.0e-10)
    for ensemblealg in (EnsembleSerial(), EnsembleThreads())
        layer = ShootingLayer(
            problem, Tsit5(),
            Tunable(1; kind = Val(:variable), default = 1.0) => [0.0, 1.0, 2.0, 2.5],
            Constant(2; kind = Val(:variable), default = 2.0) => [1.0],
            Tunable(1; default = 0.2) => [0.75, 1.75],
            Constant(2; default = 0.3) => [0.75, 1.75]; ensemblealg
        )
        ps, st = LuxCore.setup(MersenneTwister(1), layer)
        ps, st = add_parameter_segments!(layer, ps, st)
        for parameters in (ps, ComponentArray(ps), (; parameter = ps.parameter, initialcondition = ps.initialcondition))
            model = CorleoneDynamicOptProblem(layer, traj -> last(traj.u)[1], traj -> last(traj.u); ps = parameters, st, lcons = [-Inf, -Inf], ucons = [Inf, Inf])
            before = deepcopy((parameters, st))
            pattern = constraint_jacobian_sparsity(model)
            @test pattern isa SparseMatrixCSC{Bool}
            @test (parameters, st) == before
            q, rebuild = pack(default_vectorizer(), parameters)
            ids = rebuild(collect(eachindex(q)))
            nodes = ids.initialcondition.initialconditions.u₁.parameter.ps
            control = ids.parameter.parameters.p₁.parameter.p₁.ps
            @test size(pattern) == (6, length(q))
            @test all(pattern[1:2, :]) # Arbitrary user constraints remain dense.
            @test pattern[3, nodes[1]] # Free global-start assignment affects the first endpoint.
            @test !pattern[4, nodes[1]] # Full reset at 1 erases that dependency.
            @test pattern[5, nodes[2]] # Partial reset at 2 retains dependence through state 2.
            inserted = only(st.parameter.parameters.p₁.lookup.inserted)
            @test findall(pattern[end, :]) == sort(control[[inserted - 1, inserted]])
            for shift in (0.0, 0.01, -0.03)
                input = q .+ shift .* cos.(eachindex(q))
                J = ForwardDiff.jacobian(x -> optimization_constraints(model, rebuild(x), st), input)
                @test maximum(abs, J[.!Matrix(pattern)]; init = 0.0) < 1.0e-9
            end
            changed = merge(st, (; timestops = sort(vcat(st.timestops, 0.25))))
            @test_throws ArgumentError constraint_jacobian_sparsity(model; st = changed)
        end
    end
end

@testset "Empty constraint patterns" begin
    problem = ODEProblem((u, p, t) -> p, [1.0], (0.0, 1.0), [0.2])
    layer = ShootingLayer(problem, Tsit5())
    model = CorleoneDynamicOptProblem(layer, traj -> last(traj.u)[1])
    @test size(constraint_jacobian_sparsity(model)) == (0, 1)
    fixed = ShootingLayer(problem, Tsit5(); fixed_parameters = [1])
    model = CorleoneDynamicOptProblem(fixed, traj -> last(traj.u)[1], traj -> last(traj.u)[1]; lcons = [0.0], ucons = [Inf])
    @test size(constraint_jacobian_sparsity(model)) == (1, 0)
end

@testset "Noncontiguous shared reset groups and terminal assignments" begin
    problem = ODEProblem((u, p, t) -> [p[1] * sum(u), u[1] + p[2], u[2], u[3]], ones(4), (0.0, 3.0), [0.1, 0.2]; abstol = 1.0e-10, reltol = 1.0e-10)
    layer = ShootingLayer(
        problem, Tsit5(),
        Tunable([3, 1]; kind = Val(:variable), default = [0.5, 0.4]) => [0.0, 1.0, 2.0, 3.0],
        Constant(4; kind = Val(:variable), default = 1.0) => [0.0, 1.0, 2.0, 3.0],
        Tunable([2, 1]; default = [0.2, 0.1]) => nothing
    )
    model = CorleoneDynamicOptProblem(layer, traj -> last(traj.u)[1])
    pattern = constraint_jacobian_sparsity(model)
    q, rebuild = pack(default_vectorizer(), model.ps)
    ids = rebuild(collect(eachindex(q)))
    nodes = ids.initialcondition.initialconditions.u₁.parameter
    @test size(pattern) == (4, length(q))
    @test pattern[1, nodes.u₁.ps[2]]
    @test !pattern[1, nodes.u₃.ps[2]]
    @test pattern[2, nodes.u₃.ps[2]]
    @test !pattern[2, nodes.u₁.ps[2]]
    @test !any(pattern[:, [nodes.u₁.ps[end], nodes.u₃.ps[end]]])
    for shift in (0.0, 0.03)
        x = q .+ shift
        J = ForwardDiff.jacobian(x -> optimization_constraints(model, rebuild(x)), x)
        @test maximum(abs, J[.!Matrix(pattern)]; init = 0.0) < 1.0e-9
    end
end

@testset "Parsed expression rows and saved sides" begin
    problem = ODEProblem((u, p, t) -> [p[1]], [1.0], (0.0, 3.0), [0.2, 0.3]; saveat = [0.25, 0.75, 1.25, 1.75, 2.25, 2.75], abstol = 1.0e-10, reltol = 1.0e-10)
    layer = ShootingLayer(
        problem, Tsit5(), Tunable(1; kind = Val(:variable), default = 1.0) => [0.0, 1.0, 2.0, 3.0],
        Tunable(1; default = 0.2) => [0.5, 1.5, 2.5, 3.0], Constant(2; default = 0.3) => nothing
    )
    for evaluator in (Val(:Eval), default_evaluator())
        model = CorleoneDynamicOptProblem(
            layer, :(u₁(3.0)), :(u₁(1.0)), :([p₁(0.5), u₁(0.5), u₁(1.25), p₂(2.0), 7.0]),
            :(sum(u₁)), :(sum(p₁)), :(identity([u₁(1.0), p₁(2.0)])),
            :([[u₁(1.0)]..., p₁(2.0)]), :(ifelse(p₁(0.25) > 0, u₁(1.0), u₁(2.0))), traj -> last(traj.u)[1];
            evaluator, lcons = [0.0, zeros(5), 0.0, 0.0, zeros(2), zeros(2), 0.0, 0.0],
            ucons = [Inf, fill(Inf, 5), Inf, Inf, fill(Inf, 2), fill(Inf, 2), Inf, Inf]
        )
        q, rebuild = pack(default_vectorizer(), model.ps)
        ids = rebuild(collect(eachindex(q)))
        nodes = ids.initialcondition.initialconditions.u₁.parameter.ps
        control = ids.parameter.parameters.p₁.parameter.ps
        pattern = constraint_jacobian_sparsity(model)
        @test findall(pattern[1, :]) == [nodes[2]]
        @test findall(pattern[2, :]) == [control[2]] # Right-continuous control.
        @test findall(pattern[3, :]) == sort([nodes[1], control[1]]) # State at switch precedes its effect.
        @test findall(pattern[4, :]) == sort([nodes[2], control[2]]) # Internal saved sample.
        @test !any(pattern[5:6, :]) # Fixed parameter and constant expression.
        @test !pattern[7, nodes[end]] && !pattern[7, control[end]]
        @test all(pattern[7, vcat(nodes[1:3], control[1:4])]) # Includes incoming sides before resets.
        @test findall(pattern[8, :]) == collect(control) # Parameter series includes the final switch.
        for row in 9:12
            @test findall(pattern[row, :]) == sort([nodes[2], control[3]]) # Nonliteral and splatted vectors use a union.
        end
        @test findall(pattern[13, :]) == sort([nodes[2], nodes[3], control[1]])
        @test all(pattern[14, :]) # Opaque callback fallback.
        for shift in (0.0, 0.03, -0.4)
            input = q .+ shift
            J = ForwardDiff.jacobian(x -> optimization_constraints(model, rebuild(x)), input)
            @test maximum(abs, J[.!Matrix(pattern)]; init = 0.0) < 1.0e-9
        end
    end
end

@testset "Callback initialization keeps boundary reads conservative" begin
    initialize = (callback, u, t, integrator) -> (integrator.u[1] = integrator.u[2] + integrator.p[1])
    callback = Corleone.SciMLBase.DiscreteCallback((u, t, integrator) -> false, integrator -> nothing; initialize)
    problem = ODEProblem((u, p, t) -> zero.(u), [1.0, 2.0], (0.0, 2.0), [0.2]; callback)
    layer = ShootingLayer(problem, Tsit5(), Tunable([1, 2]; kind = Val(:variable), default = [1.0, 2.0]) => [1.0], Tunable(1; default = 0.2) => [0.5])
    model = CorleoneDynamicOptProblem(layer, :(u₁(2.0)), :(u₁(1.0)); lcons = [0.0], ucons = [Inf])
    pattern = constraint_jacobian_sparsity(model)
    q, rebuild = pack(default_vectorizer(), model.ps)
    ids = rebuild(collect(eachindex(q)))
    @test pattern[1, ids.parameter.parameters.p₁.parameter.ps[2]]
    J = ForwardDiff.jacobian(x -> optimization_constraints(model, rebuild(x)), q)
    @test maximum(abs, J[.!Matrix(pattern)]; init = 0.0) < 1.0e-9
end

@testset "Parsed noncontiguous parameter groups" begin
    problem = ODEProblem((u, p, t) -> [sum(p)], [1.0], (0.0, 2.0), [0.1, 0.2, 0.3])
    layer = ShootingLayer(problem, Tsit5(), Tunable([3, 1]; default = [0.3, 0.1]) => [1.0], Constant(2; default = 0.2) => nothing)
    model = CorleoneDynamicOptProblem(layer, :(u₁(2.0)), :([p₃(0.5), p₁(1.0), p₂(1.0)]); lcons = zeros(3), ucons = fill(Inf, 3))
    pattern = constraint_jacobian_sparsity(model)
    q, rebuild = pack(default_vectorizer(), model.ps)
    ids = rebuild(collect(eachindex(q))).parameter.parameters.p₁.parameter
    @test findall(pattern[1, :]) == [ids.p₃.ps[1]]
    @test findall(pattern[2, :]) == [ids.p₁.ps[2]]
    @test !any(pattern[3, :])
    expected = ForwardDiff.jacobian(x -> optimization_constraints(model, rebuild(x)), q)
    @test maximum(abs, expected[.!Matrix(pattern)]; init = 0.0) < 1.0e-9
end

@testset "Parameter signal dependencies follow the supplied state permutation" begin
    problem = ODEProblem((u, p, t) -> [sum(p)], [1.0], (0.0, 2.0), [0.1, 0.2, 0.3])
    layer = ShootingLayer(problem, Tsit5(), Tunable([3, 1]; default = [0.3, 0.1]) => [1.0], Constant(2; default = 0.2) => nothing)
    ps, st = LuxCore.setup(MersenneTwister(1), layer)
    changed = merge(st, (; parameter = merge(st.parameter, (; permutation = reverse(st.parameter.permutation)))))
    model = CorleoneDynamicOptProblem(layer, :(u₁(2.0)), :([p₃(0.5), p₁(1.0), p₂(1.0)]); ps, st = changed, lcons = zeros(3), ucons = fill(Inf, 3))
    q, rebuild = pack(default_vectorizer(), ps)
    pattern = constraint_jacobian_sparsity(model)
    J = ForwardDiff.jacobian(x -> optimization_constraints(model, rebuild(x), changed), q)
    @test maximum(abs, J[.!Matrix(pattern)]; init = 0.0) < 1.0e-9
end

@testset "Parameter series exclude intervals outside the saved span" begin
    problem = ODEProblem((u, p, t) -> p, [1.0], (0.0, 2.0), [0.2]; saveat = 0.0:0.1:2.0)
    layer = ShootingLayer(problem, Tsit5(), Tunable(1; default = [0.1, 0.2, 0.3, 0.4]) => [-1.0, 0.5, 3.0])
    model = CorleoneDynamicOptProblem(layer, :(u₁(2.0)), :(sum(p₁)); lcons = [0.0], ucons = [Inf])
    q, rebuild = pack(default_vectorizer(), model.ps)
    ids = rebuild(collect(eachindex(q))).parameter.parameters.p₁.parameter.ps
    pattern = constraint_jacobian_sparsity(model)
    @test findall(pattern[1, :]) == collect(ids[2:3])
    J = ForwardDiff.jacobian(x -> optimization_constraints(model, rebuild(x)), q)
    @test maximum(abs, J[.!Matrix(pattern)]; init = 0.0) < 1.0e-9
end

@testset "Rebuilt expression problems keep their types" begin
    problem = ODEProblem((u, p, t) -> [p[1] * u[1]], [1.0], (0.0, 1.0), [0.2]; saveat = 0.25)
    layer = ShootingLayer(problem, Tsit5(), Tunable(1; default = 0.2) => [0.5])
    for evaluator in (Val(:Eval), default_evaluator())
        build() = CorleoneDynamicOptProblem(
            layer, :(u₁(1.0)^2), :(identity(u₁)), :(u₁(0.5) + p₁(0.75));
            evaluator, lcons = [zeros(5), 0.0], ucons = [fill(Inf, 5), Inf]
        )
        first_model, second_model = build(), build()
        @test typeof(first_model.constraints) === typeof(second_model.constraints)
        @test typeof(first_model.objective) === typeof(second_model.objective)
        @test typeof(first_model) === typeof(second_model)
        optimization_constraints(first_model)
        count_specializations() = sum(m -> count(!isnothing, Base.specializations(m)), methods(optimization_constraints))
        before = count_specializations()
        @test optimization_constraints(second_model) == optimization_constraints(first_model)
        @test count_specializations() == before # The rebuilt problem compiles nothing new.
    end
    # Distinct expressions remain distinct functions.
    a = CorleoneDynamicOptProblem(layer, :(u₁(1.0)^2))
    b = CorleoneDynamicOptProblem(layer, :(u₁(1.0)^3))
    @test typeof(a.objective) !== typeof(b.objective)
    @test optimization_objective(a) ≈ optimization_objective(b)^(2 / 3)
end

@testset "Elementwise series expression classification" begin
    indices = Dict(:u₁ => (:states, 1), :u₂ => (:states, 2), :p₁ => (:parameters, 1))
    classify(expr) = Corleone.elementwise_series_expression(expr, indices)
    for expr in (:u₁, :p₁, :(identity(u₁)), :(u₁ .- 0.5 .* u₂(1.0)), :(sin.(u₁) .+ p₁), :(.-u₁), :(ifelse.(u₁ .> 0.0, u₂, 1.0)), :(identity(u₁ .^ 2)))
        @test classify(expr) === :series
    end
    for expr in (:(cumsum(u₁)), :(sum(u₁)), :(u₁ * 2.0), :(u₁ .+ sum(u₂)), :(u₁ .+ c), :(u₁(1.0) .+ 1.0), :(u₁[1:2] .+ 0.0), :(f.(u₁...)), :(u₁ .+ [1.0, 2.0]), :(identity(u₁(1.0))))
        @test classify(expr) !== :series
    end
    @test classify(:(u₁ .+ $([1.0, 2.0]))) === :series # Interpolated constants are allowed.
end

@testset "Saved series rows depend on their own sample" begin
    function lotka!(du, u, p, t)
        du[1] = u[1] - p[1] * u[1] * u[2] - 0.4 * p[2] * u[1]
        du[2] = -u[2] + p[1] * u[1] * u[2] - 0.2 * p[2] * u[2]
        du[3] = (u[1] - 1.0)^2 + (u[2] - 1.0)^2
        return nothing
    end
    problem = ODEProblem(lotka!, [0.5, 0.7, 0.0], (0.0, 3.0), [1.0, 0.3]; saveat = 0.25, abstol = 1.0e-10, reltol = 1.0e-10)
    control = Tunable(2; default = 0.3, lowerbound = 0.0, upperbound = 1.0) => collect(0.5:0.5:2.5)
    fixed = Constant(1) => nothing
    nodes = [1.0, 2.0]
    for shooting in (:single, :multiple)
        lifted = shooting === :multiple ? (Tunable([1, 2]; kind = Val(:variable)) => nodes,) : ()
        layer = ShootingLayer(problem, Tsit5(), control, fixed, lifted...; controls = [2], quadratures = [3])
        ps, st = LuxCore.setup(MersenneTwister(1), layer)
        ps, st = add_parameter_segments!(layer, ps, st)
        nsamples = 13 + (shooting === :multiple ? length(nodes) : 0) # Both sides of saved resets.
        expressions = (:(identity(u₁)), :(u₂ .* u₁(3.0) .- p₂), :(sin.(u₁) .+ 1.0), :u₂, :(cumsum(u₁))) # A bare signal is its series.
        model = CorleoneDynamicOptProblem(
            layer, :(u₃(3.0)), expressions...; ps, st,
            lcons = [fill(-Inf, nsamples) for _ in expressions], ucons = [fill(Inf, nsamples) for _ in expressions]
        )
        @test length(model.times) == nsamples
        q, rebuild = pack(default_vectorizer(), ps)
        ids = rebuild(collect(eachindex(q)))
        controls = ids.parameter.parameters.p₂.parameter.ps
        pattern = constraint_jacobian_sparsity(model)
        rows(j) = ((j - 1) * nsamples + 1):(j * nsamples)
        path, timed, broadcast, bare, cumulative = map(rows, 1:5)
        union_columns = reduce(.|, eachrow(pattern[path, :]))
        @test all(row -> pattern[row, :] == union_columns, cumulative) # Non-elementwise stays dense.
        @test count(pattern[path, :]) < length(path) * count(union_columns) # Strictly sparser.
        @test !any(pattern[first(path), :]) # The fixed initial value has no decisions.
        @test findall(pattern[path[2], :]) == [controls[1]] # t = 0.25 precedes later controls.
        @test pattern[broadcast, :] == pattern[path, :]
        @test !any(pattern[path[2], controls[2:end]])
        terminal = findall(pattern[last(path), :])
        for k in eachindex(path)
            @test findall(pattern[timed[k], :]) == sort(union(findall(pattern[path[k], :]), terminal, findall(pattern[bare[k], :]), [controls[searchsortedlast([0.5, 1.0, 1.5, 2.0, 2.5], model.times[k]) + 1]]))
        end
        if shooting === :multiple
            node = ids.initialcondition.initialconditions.u₁.parameter
            incoming, outgoing = findall(==(1.0), model.times)
            @test outgoing == incoming + 1
            @test findall(pattern[path[outgoing], :]) == [node.u₁.ps[1]] # Reset side reads only its node.
            @test findall(pattern[bare[outgoing], :]) == [node.u₂.ps[1]]
            @test !pattern[path[incoming], node.u₁.ps[1]]
            @test findall(pattern[path[incoming], :]) == sort(controls[1:2])
            @test pattern[path[outgoing + 1], node.u₂.ps[1]] # Coupled states after integration.
        end
        for shift in (0.0, 0.01, -0.03)
            input = q .+ shift .* cos.(eachindex(q))
            J = ForwardDiff.jacobian(x -> optimization_constraints(model, rebuild(x), st), input)
            @test maximum(abs, J[.!Matrix(pattern)]; init = 0.0) < 1.0e-9
        end
    end
end
