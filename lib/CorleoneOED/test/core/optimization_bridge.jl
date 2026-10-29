using CorleoneOED, LuxCore, OrdinaryDiffEqTsit5, Random, ForwardDiff, LinearAlgebra, Test

# A native NamedTuple vectorizer exercises the contract independently of
# ComponentArrays and of the Optimization.jl solver frontend.
numeric_length(x::AbstractArray) = length(x)
numeric_length(x::NamedTuple) = sum(numeric_length, values(x); init = 0)
flatten_native(x::AbstractArray) = vec(x)
flatten_native(x::NamedTuple) = reduce(vcat, map(flatten_native, values(x)); init = Float64[])
restore_native(template::AbstractArray, q) = reshape(q, size(template))
function restore_native(template::NamedTuple, q)
    members = values(template)
    restored = map(ntuple(identity, length(members))) do i
        start = 1 + sum(numeric_length, members[1:(i - 1)]; init = 0)
        restore_native(members[i], view(q, start:(start + numeric_length(members[i]) - 1)))
    end
    return NamedTuple{keys(template)}(restored)
end
function Corleone.pack(::Val{:NativeOEDTest}, ps::NamedTuple)
    return collect(flatten_native(ps)), q -> restore_native(ps, q)
end

struct TraceCriterionForBridge <: AbstractCriterion end
(::TraceCriterionForBridge)(F::Symmetric) = tr(F)

problem = ODEProblem((u, p, t) -> p[1] .* u, [1.0], (0.0, 1.0), [-2.0]; abstol = 1.0e-11, reltol = 1.0e-11)
parameter = Tunable(1; default = -2.0, lowerbound = -3.0, upperbound = -1.0) => nothing
measurement = (
    ContinuousMeasurement(:density, [0.25, 0.75], (u, p, t) -> u[1]),
    DiscreteMeasurement(:events, [0.0, 0.5, 1.0], (u, p, t) -> u[1]),
)
layer = OEDLayer(problem, Tsit5(), parameter; measurements = measurement)

@testset "OED uses the core bridge with a native vectorizer" begin
    @test isnothing(Base.get_extension(CorleoneOED, :CorleoneOEDOptimizationExtension))
    for design in (layer, MultiExperimentLayer(layer, 2))
        ps, st = LuxCore.setup(Xoshiro(1), design)
        n = design isa OEDLayer ? 1 : 2
        budgets = repeat([0.5, 1.0], n)
        opt = CorleoneOED.SciMLBase.OptimizationProblem(
            design, TraceCriterionForBridge();
            ps, st, M = budgets, vectorize = Val(:NativeOEDTest), integer_weights = true
        )
        @test sum(opt.int) == 6n
        @test length(opt.int) == length(opt.u0) == 7n
        @test opt.ucons == budgets
        @test opt.lcons == zeros(2n)
        @test all(opt.lb .<= opt.u0 .<= opt.ub)
        q, reconstruct = pack(Val(:NativeOEDTest), ps)
        @test opt.f(q, st) ≈ tr(first(fisher_information(design, nothing, ps, st)))
        loss(q) = opt.f(q, st)
        gradient = ForwardDiff.gradient(loss, q)
        @test all(isfinite, gradient)
        finite = [(loss(q + 1.0e-5 * [j == i for j in eachindex(q)]) - loss(q - 1.0e-5 * [j == i for j in eachindex(q)])) / 2.0e-5 for i in eachindex(q)]
        @test gradient ≈ finite rtol = 1.0e-5
        updated = merge(st, (; F_init = 2one(st.F_init)))
        @test opt.f(q, updated) ≈ opt.f(q, st) + 2size(st.F_init, 1)
        buffer = zeros(2n)
        opt.f.cons(buffer, q, st)
        @test buffer ≈ sampling_sums(design, nothing, ps, st)
    end
end

@testset "Multiple experiments share callback preparation and layout checks" begin
    multi = MultiExperimentLayer(layer, 2)
    ps, st = LuxCore.setup(Xoshiro(2), multi)
    scalar = trajectories -> last(trajectories.experiment_1.u)[1]
    inplace! = (res, trajectories) -> (res[1] = last(trajectories.experiment_2.u)[1]; :ignored)
    model = CorleoneDynamicOptProblem(
        multi, TraceCriterionForBridge(), scalar, inplace!;
        ps, st, M = fill(1.0, 4), lcons = [0.0, [0.0]], ucons = [1.0, [1.0]]
    )
    values = optimization_constraints(model)
    @test length(values) == 6
    @test values[1:2] ≈ fill(exp(-2), 2)
    @test values[3:end] ≈ [1.0, 3.0, 1.0, 3.0]
    buffer = similar(values)
    @test optimization_constraints!(buffer, model) ≈ values
    changed = deepcopy(st)
    push!(changed.experiments.experiment_1.timestops, 0.125)
    @test_throws ArgumentError optimization_objective(model, ps, changed)
    @test_throws ArgumentError CorleoneDynamicOptProblem(multi, :(u₁(1.0)); ps, st)
end
