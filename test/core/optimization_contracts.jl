using Corleone, LuxCore, OrdinaryDiffEqTsit5, Random, Test, ComponentArrays

@testset "Optimization bounds follow tunable parameter structure" begin
    calls = Ref(0)
    initialize = (rng, n::Int) -> begin
        calls[] += 1
        fill(1.0, n)
    end
    prob = ODEProblem((u, p, t) -> zero(u), [1.0, 2.0, 3.0], (0.0, 3.0), [1.0, 2.0, 3.0])
    layer = ShootingLayer(
        prob, Tsit5(),
        Tunable(1; kind = Val(:variable), default = initialize, lowerbound = 0.0, upperbound = 4.0) => [0.0, 1.0],
        Constant(2; kind = Val(:variable), lowerbound = 0.0, upperbound = 10.0) => [1.0],
        Tunable(1; kind = Val(:parameter), default = initialize, lowerbound = -1.0, upperbound = 2.0) => [1.0, 2.0],
        Constant(2; kind = Val(:parameter), lowerbound = 0.0, upperbound = 5.0) => nothing
    )
    ps, st = LuxCore.setup(MersenneTwister(1), layer)
    @test calls[] == 2
    lower, upper = get_bounds(layer, ps, st)
    @test calls[] == 2
    @test keys(lower) == keys(upper) == keys(ps)
    @test lower.initialcondition.initialconditions.u₁.parameter.ps == [0.0, 0.0]
    @test upper.initialcondition.initialconditions.u₁.parameter.ps == [4.0, 4.0]
    @test lower.initialcondition.initialconditions.u₂ == (;)
    @test lower.initialcondition.initialconditions.u₃ == (;)
    @test lower.parameter.parameters.p₁.parameter.ps == fill(-1.0, 3)
    @test upper.parameter.parameters.p₁.parameter.ps == fill(2.0, 3)
    @test lower.parameter.parameters.p₂ == (;)
    @test lower.parameter.parameters.p₃.parameter.ps == [-Inf]
    @test upper.parameter.parameters.p₃.parameter.ps == [Inf]
    @test getaxes(ComponentArray(lower)) == getaxes(ComponentArray(ps))
    @test getaxes(ComponentArray(upper)) == getaxes(ComponentArray(ps))
    @test length(ComponentArray(lower)) == length(ComponentArray(ps)) == 6
    @test get_bounds(layer, ComponentArray(ps), st) == (lower, upper)
    @test get_bounds(layer, ps) == (lower, upper)
    @test lower.initialcondition.initialconditions.u₁.parameter.ps !== ps.initialcondition.initialconditions.u₁.parameter.ps
    @test ps.initialcondition.initialconditions.u₁.parameter.ps == ones(2)
    # Optimizer ordering follows the supplied parameter tree, not symbol names
    # or the layer's constructor order.
    reversed = (;
        parameter = (;
            parameters = (;
                p₃ = ps.parameter.parameters.p₃,
                p₁ = ps.parameter.parameters.p₁,
                p₂ = ps.parameter.parameters.p₂,
            ),
        ), initialcondition = ps.initialcondition,
    )
    reordered_lower, reordered_upper = get_bounds(layer, reversed, st)
    @test keys(reordered_lower) == keys(reordered_upper) == keys(reversed)
    @test keys(reordered_lower.parameter.parameters) == (:p₃, :p₁, :p₂)
    @test getaxes(ComponentArray(reordered_lower)) == getaxes(ComponentArray(reversed))
    @test getaxes(ComponentArray(reordered_upper)) == getaxes(ComponentArray(reversed))
    @test collect(ComponentArray(reordered_lower)) == [-Inf, -1.0, -1.0, -1.0, 0.0, 0.0]
    @test get_bounds(layer, ComponentArray(reversed), st) == (reordered_lower, reordered_upper)
    @test_throws ArgumentError get_bounds(layer, collect(ComponentArray(ps)), st)
    @test_throws ArgumentError get_bounds(layer, (; initialcondition = ps.initialcondition), st)
    missing = merge(ps, (; parameter = (; parameters = (; p₁ = ps.parameter.parameters.p₁))))
    @test_throws ArgumentError get_bounds(layer, missing, st)
    # Metadata updates are reflected without rerunning initialization.
    layer.sys.bounds[:p₁] = (-2.0, 3.0)
    lo, hi = get_bounds(layer, ps, st)
    @test lo.parameter.parameters.p₁.parameter.ps == fill(-2.0, 3)
    @test hi.parameter.parameters.p₁.parameter.ps == fill(3.0, 3)
    @test calls[] == 2
end

@testset "Unbounded and empty optimization layouts" begin
    prob = ODEProblem((u, p, t) -> zero(u), [1], (0.0, 1.0), Int[])
    layer = ShootingLayer(prob, Tsit5(), Tunable(1; kind = Val(:variable)) => [0.0])
    ps, st = LuxCore.setup(MersenneTwister(1), layer)
    lo, hi = get_bounds(layer, ps, st)
    @test lo.initialcondition.initialconditions.u₁.parameter.ps == [-Inf]
    @test hi.initialcondition.initialconditions.u₁.parameter.ps == [Inf]
    @test eltype(lo.initialcondition.initialconditions.u₁.parameter.ps) == Float64
    @test lo.parameter.parameters == (;)
    fixed = ShootingLayer(prob, Tsit5(), Constant(1; kind = Val(:variable)) => [0.0])
    ps, st = LuxCore.setup(MersenneTwister(1), fixed)
    lo, hi = get_bounds(fixed, ps, st)
    @test lo == hi == ps
    @test isempty(ComponentArray(lo))
end

@testset "Shooting constraint count, order, and buffers" begin
    prob = ODEProblem((u, p, t) -> [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], (0.0, 3.0), Float64[])
    for alg in (EnsembleSerial(), EnsembleThreads())
        layer = ShootingLayer(
            prob, Tsit5(),
            Tunable(1; kind = Val(:variable), default = [10.0, 20.0, 99.0]) => [0.0, 1.0, 3.0],
            Constant(2; kind = Val(:variable), default = 7.0) => [1.0, 2.0],
            Tunable(3; kind = Val(:variable), default = [30.0, 40.0]) => [1.0, 2.0]; ensemblealg = alg
        )
        ps, st = LuxCore.setup(MersenneTwister(1), layer)
        @test get_number_of_shooting_constraints(layer, ps, st) == 3
        @test get_number_of_shooting_constraints(layer, nothing, st) == 3
        trajectory, next_st = layer(nothing, ps, st)
        # t=0 assigns a free initial value; t=1 matches u1,u3; t=2 matches u3.
        @test isempty(first(trajectory.residuals))
        expected = [9.0, 24.0, 7.0]
        @test shooting_constraints(trajectory) ≈ expected
        @test length(shooting_constraints(trajectory)) == get_number_of_shooting_constraints(layer, ps, st)
        buffer = fill(NaN, 3)
        @test shooting_constraints!(buffer, trajectory) === buffer
        @test buffer ≈ expected
        storage = fill(123.0, 5)
        @test shooting_constraints!(view(storage, 2:4), trajectory) ≈ expected
        @test storage[[1, 5]] == [123.0, 123.0]
        wrong = fill(123.0, 2)
        @test_throws DimensionMismatch shooting_constraints!(wrong, trajectory)
        @test wrong == fill(123.0, 2)
        # Refreshing after moving an IC schedule changes the metadata count.
        child = st.initialcondition.initialconditions.u₃
        lookup = merge(child.lookup, (; collection = Dict(1.0 => 1, 3.0 => 2)))
        children = merge(st.initialcondition.initialconditions, (; u₃ = merge(child, (; lookup))))
        state = merge(st, (; initialcondition = merge(st.initialcondition, (; initialconditions = children))))
        state = merge(state, (; timestops = Corleone.get_timestops(layer, nothing, state)))
        @test get_number_of_shooting_constraints(layer, ps, state) == 2
        moved, _ = layer(nothing, ps, state)
        @test length(shooting_constraints(moved)) == 2
    end
    layer = ShootingLayer(prob, Tsit5())
    ps, st = LuxCore.setup(MersenneTwister(1), layer)
    trajectory, _ = layer(nothing, ps, st)
    @test get_number_of_shooting_constraints(layer, ps, st) == 0
    @test isempty(shooting_constraints(trajectory))
    empty = Float64[]
    @test shooting_constraints!(empty, trajectory) === empty
end

# These layers only implement Lux's parameter/state interface. Bounds must work
# through generic recursion, without concrete-layer bounds methods.
struct BoundsTestLeaf <: LuxCore.AbstractLuxLayer
    calls::Base.RefValue{Int}
end
function LuxCore.initialparameters(rng::Random.AbstractRNG, layer::BoundsTestLeaf)
    layer.calls[] += 1
    return (; weight = ones(Float32, 2, 2), bias = ones(Float32, 2), scale = 1.0f0)
end
LuxCore.initialstates(::Random.AbstractRNG, ::BoundsTestLeaf) = (;)
struct BoundsTestContainer{L} <: LuxCore.AbstractLuxContainerLayer{(:layers,)}
    layers::L
end
struct BoundsTestWrapper{L} <: LuxCore.AbstractLuxWrapperLayer{:layer}
    layer::L
end

@testset "Lux-style bounds recurse through generic models" begin
    calls = Ref(0)
    leaf = BoundsTestLeaf(calls)
    prob = ODEProblem((u, p, t) -> [p[1]], [1.0], (0.0, 2.0), [0.2])
    shooter = ShootingLayer(
        prob, Tsit5(),
        Tunable(1; kind = Val(:variable), lowerbound = 0.0, upperbound = 4.0) => [1.0],
        Tunable(1; kind = Val(:parameter), lowerbound = -0.5, upperbound = 0.5) => [1.0]
    )
    model = BoundsTestContainer(
        (;
            dense = leaf,
            shooting = BoundsTestWrapper(shooter),
            fixed = Corleone.FixedLayer(leaf),
            optional = nothing,
        )
    )
    ps, st = LuxCore.setup(MersenneTwister(1), model)
    @test calls[] == 2
    lower = get_lower_bound(model, ps, st)
    upper = get_upper_bound(model, ps, st)
    @test calls[] == 2
    @test get_bounds(model, ps, st) == (lower, upper)
    @test get_lower_bound(model, ps) == lower
    @test get_upper_bound(model, ps) == upper
    @test lower.layers.dense.weight == fill(-Inf, 2, 2)
    @test upper.layers.dense.bias == fill(Inf, 2)
    @test lower.layers.dense.scale == -Inf
    @test upper.layers.dense.scale == Inf
    @test lower.layers.fixed == upper.layers.fixed == (;)
    @test lower.layers.optional == upper.layers.optional == (;)
    @test lower.layers.shooting.parameter.parameters.p₁.parameter.ps == [-0.5, -0.5]
    @test upper.layers.shooting.parameter.parameters.p₁.parameter.ps == [0.5, 0.5]
    @test lower.layers.shooting.initialcondition.initialconditions.u₁.parameter.ps == [0.0]
    @test upper.layers.shooting.initialcondition.initialconditions.u₁.parameter.ps == [4.0]
    @test get_lower_bound(shooter.parameter, ps.layers.shooting.parameter) != lower.layers.shooting.parameter
    @test getaxes(ComponentArray(lower)) == getaxes(ComponentArray(ps))
    @test getaxes(ComponentArray(upper)) == getaxes(ComponentArray(ps))
    @test get_bounds(model, ComponentArray(ps), st) == (lower, upper)
    @test calls[] == 2

    # Collections preserve their nesting and field order, including wrappers
    # whose parameter/state shape is the inner layer's shape without another key.
    @test get_bounds(model.layers, ps.layers, st.layers) == (lower.layers, upper.layers)
    @test get_bounds(BoundsTestWrapper(model), ps, st) == (lower, upper)
    pair = (leaf, shooter)
    pair_ps = (ps.layers.dense, ps.layers.shooting)
    pair_st = (st.layers.dense, st.layers.shooting)
    pair_lower, pair_upper = get_bounds(pair, pair_ps, pair_st)
    @test pair_lower isa Tuple
    @test pair_lower == (lower.layers.dense, lower.layers.shooting)
    @test pair_upper == (upper.layers.dense, upper.layers.shooting)
    @test_throws ArgumentError get_lower_bound(pair, (first(pair_ps),))
    @test_throws ArgumentError get_upper_bound(model.layers, (; dense = ps.layers.dense))
    @test calls[] == 2

    # Preserve named zero-length parameter fields through transparent wrappers,
    # even when a ComponentArray's flattened storage is empty.
    empty_prob = ODEProblem((u, p, t) -> zero(u), [1.0], (0.0, 1.0), Float64[])
    empty_shooter = ShootingLayer(
        empty_prob, Tsit5(),
        Tunable(1; kind = Val(:variable)) => Float64[]
    )
    empty_ps, empty_st = LuxCore.setup(MersenneTwister(1), empty_shooter)
    empty_lower, empty_upper = get_bounds(empty_shooter, empty_ps, empty_st)
    @test empty_lower.initialcondition.initialconditions.u₁.parameter.ps == Float64[]
    @test get_bounds(empty_shooter, ComponentArray(empty_ps), empty_st) == (empty_lower, empty_upper)
    @test get_bounds(BoundsTestWrapper(empty_shooter), ComponentArray(empty_ps), empty_st) == (empty_lower, empty_upper)

    collection = reshape([leaf, leaf], 1, 2)
    collection_ps = reshape([ps.layers.dense, ps.layers.dense], 1, 2)
    lo, hi = get_bounds(collection, collection_ps)
    @test size(lo) == size(hi) == (1, 2)
    @test all(==(lower.layers.dense), lo)
    @test all(==(upper.layers.dense), hi)
    @test calls[] == 2
end
