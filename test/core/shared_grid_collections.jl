using Corleone, LuxCore, Random, OrdinaryDiffEqTsit5, ComponentArrays, Test
using ForwardDiff, ReverseDiff, Zygote, DiffEqBase
import SciMLBase

function shared_grid_rhs!(du, u, p, t)
    du .= p
    return nothing
end

@testset "Scalar buffers share lookups across equal grids" begin
    prob = ODEProblem(
        ODEFunction{true, SciMLBase.FullSpecialize}(shared_grid_rhs!),
        zeros(11), (0.0, 3.0), ones(11);
        sensealg = SensitivityADPassThrough(), abstol = 1.0e-10, reltol = 1.0e-10
    )
    others = setdiff(1:11, [2, 11])
    control_others = setdiff(1:11, [2, 5, 11])
    definitions = (
        Tunable([11, 2]; kind = Val(:variable), default = [1.1, 0.2], upperbound = [11.0, 2.0]) => [1.0, 2.0],
        Constant(others; kind = Val(:variable), default = zeros(length(others))) => [0.75, 1.25, 2.0],
        Tunable([11, 2]; default = [1.1, 0.2], upperbound = [2.0, 4.0]) => [0.5, 1.5],
        Constant(5; default = [0.5, 0.6, 0.7]) => copy([0.5, 1.5]),
    )
    layer = ShootingLayer(prob, Tsit5(), definitions...; fixed_parameters = control_others)
    ps, st = LuxCore.setup(MersenneTwister(1), layer)
    @test length(layer.initialcondition.initialconditions) == 2
    @test length(layer.parameter.parameters) == 2
    @test keys(layer.parameter.parameters.p₂.parameter) == (:p₂, :p₅, :p₁₁)
    @test layer.parameter.parameters.p₂.parameter.p₅ isa Corleone.FixedLayer
    @test ps.parameter.parameters.p₂.parameter.p₂.ps == fill(0.2, 3)
    @test ps.parameter.parameters.p₂.parameter.p₁₁.ps == fill(1.1, 3)
    @test all(value -> value isa Number, ps.parameter.parameters.p₂.parameter.p₂.ps)
    @test ps.initialcondition.initialconditions.u₂.parameter.u₂.ps == [0.2, 0.2]
    @test ps.initialcondition.initialconditions.u₂.parameter.u₁₁.ps == [1.1, 1.1]
    @test first(layer.parameter(1.0, ps.parameter, st.parameter))[[2, 5, 11]] == [0.2, 0.6, 1.1]
    ((u, residuals), _) = layer.initialcondition((zeros(11), 1.0), ps.initialcondition, st.initialcondition)
    @test u[[2, 11]] == [0.2, 1.1]
    @test residuals == [0.2, 1.1]
    @test st.initialcondition.active == [i in (2, 11) for i in 1:11]
    @test get_number_of_shooting_constraints(layer, ps, st) == 4
    ps, st = add_parameter_segments!(layer, ps, st)
    @test st.parameter.parameters.p₂.lookup.collection == [0.5, 1.5, 2.0]
    @test st.parameter.parameters.p₂.lookup.inserted == [4]
    @test st.parameter.parameters.p₂.parameter.p₅.parameters.ps == [0.5, 0.6, 0.7, 0.7]
    @test st.parameter.parameters.p₁.lookup.collection === nothing
    @test get_number_of_shooting_constraints(layer, ps, st) == 6
    @test first(layer.parameter(2.5, ps.parameter, st.parameter))[[2, 5, 11]] == [0.2, 0.7, 1.1]
    lower, upper = get_bounds(layer, ps, st)
    @test upper.parameter.parameters.p₂.parameter.p₂.ps == fill(4.0, 4)
    @test upper.parameter.parameters.p₂.parameter.p₁₁.ps == fill(2.0, 4)
    @test upper.parameter.parameters.p₂.parameter.p₅ == (;)
    @test upper.initialcondition.initialconditions.u₂.parameter.u₂.ps == fill(2.0, 2)
    q = ComponentArray(ps)
    axes = getaxes(q)
    function packing(q)
        params = ComponentArray(q, axes)
        ((state, residuals), _) = layer.initialcondition((zeros(11), 1.0), params.initialcondition, st.initialcondition)
        parameters, _ = layer.parameter(2.5, params.parameter, st.parameter)
        return sum(abs2, state) + sum(abs2, residuals) + sum(abs2, parameters)
    end
    function loss(q)
        prediction, _ = layer(nothing, ComponentArray(q, axes), st)
        return sum(abs2, last(prediction.u)) + 0.1sum(abs2, shooting_constraints(prediction))
    end
    for input in (collect(q), collect(q) .+ 0.01)
        @test ReverseDiff.gradient(packing, input) ≈ ForwardDiff.gradient(packing, input)
        @test only(Zygote.gradient(packing, input)) ≈ ForwardDiff.gradient(packing, input)
        @test ReverseDiff.gradient(loss, input) ≈ ForwardDiff.gradient(loss, input) rtol = 1.0e-6 atol = 1.0e-7
    end
    trajectory, returned = layer(nothing, ps, st)
    @test returned.initialcondition.permutation === st.initialcondition.permutation
    @test length(shooting_constraints(trajectory)) == 6
    buffer = zeros(6)
    shooting_constraints!(buffer, trajectory)
    @test buffer ≈ shooting_constraints(trajectory)
    changed = deepcopy(ps)
    changed.parameter.parameters.p₂.parameter.p₂.ps[4] += 2.0
    changed.parameter.parameters.p₂.parameter.p₁₁.ps[4] += 11.0
    changed_trajectory, _ = layer(nothing, changed, st)
    @test shooting_constraints(changed_trajectory)[5:6] ≈ [2.0, 11.0]
    threaded = ShootingLayer(prob, Tsit5(), definitions...; fixed_parameters = control_others, ensemblealg = EnsembleThreads())
    actual, _ = threaded(nothing, ps, st)
    @test actual.u ≈ trajectory.u
    @test shooting_constraints(actual) ≈ shooting_constraints(trajectory)
    threaded_loss(q) = sum(abs2, last(first(threaded(nothing, ComponentArray(q, axes), st)).u))
    serial_loss(q) = sum(abs2, last(first(layer(nothing, ComponentArray(q, axes), st)).u))
    @test ForwardDiff.gradient(threaded_loss, collect(q)) ≈ ForwardDiff.gradient(serial_loss, collect(q))
    initialized = deepcopy(ps)
    constant_initialization(MersenneTwister(1), layer, initialized, st; u0 = fill(0.1, 11), variables = [2])
    @test initialized.initialcondition.initialconditions.u₂.parameter.u₂.ps == [0.1, 0.1]
    @test initialized.initialcondition.initialconditions.u₂.parameter.u₁₁.ps == [1.1, 1.1]
    forward_initialization(MersenneTwister(1), layer, initialized, st)
    @test maximum(abs, shooting_constraints(first(layer(nothing, initialized, st)))) < 1.0e-8
end

@testset "Different scalar buffer types preserve output inference" begin
    prob = ODEProblem((u, p, t) -> p, zeros(2), (0.0, 2.0), ones(2))
    layer = ShootingLayer(
        prob, Tsit5(), Tunable(1; default = 1) => nothing,
        Tunable(2; default = 0.5) => [1.0]
    )
    ps, st = LuxCore.setup(MersenneTwister(1), layer)
    @test eltype(ps.parameter.parameters.p₁.parameter.ps) == Int
    @test eltype(ps.parameter.parameters.p₂.parameter.ps) == Float64
    output, _ = @inferred layer.parameter(0.5, ps.parameter, st.parameter)
    @test output == [1.0, 0.5]
    @test output isa Vector{Float64}
end
