using Corleone, SymbolicIndexingInterface, OrdinaryDiffEqTsit5, LuxCore, Random, Test

struct MetadataProvider{S}
    sys::S
end
SymbolicIndexingInterface.symbolic_container(provider::MetadataProvider) = provider.sys

@testset "Symbolic bookkeeping preserves the index provider" begin
    parent = SymbolCache(Dict(:cost => 3, :y => 2, :x => 1),
        Dict(:rate => 2, :input => 1), :t;
        defaults = Dict(:x => 1.0, :y => 2.0, :cost => 0.0, :input => 0.5, :rate => 1.0),
        timeseries_parameters = Dict(:input => ParameterTimeseriesIndex(1, 1)))
    limits = Dict(:input => (0.0, 1.0), :x => (-2.0, Inf))
    cache = ControlSymbolCache(parent; controls = [1], lagrange_variables = [:cost], bounds = limits)
    @test symbolic_container(cache) === parent
    @test cache.sys === parent
    @test control_symbols(cache) == [:input]
    @test control_indices(cache) == [1]
    @test lagrange_symbols(cache) == [:cost]
    @test lagrange_indices(cache) == [3]
    # SII's standard queries forward; bookkeeping is specific to the cache.
    provider = MetadataProvider(cache)
    for query in (control_symbols, control_indices, lagrange_symbols, lagrange_indices)
        @test_throws MethodError query(parent)
        @test_throws MethodError query(provider)
    end
    @test_throws MethodError Corleone.get_bounds(provider, :input)
    @test parameter_index(provider, :input) == 1
    @test variable_index(provider, :cost) == 3
    @test variable_symbols(cache) == variable_symbols(parent)
    @test parameter_symbols(cache) == parameter_symbols(parent)
    @test all_variable_symbols(cache) == all_variable_symbols(parent)
    @test all_symbols(cache) == all_symbols(parent)
    @test default_values(cache) === default_values(parent)
    @test default_bounds(cache) === limits
    @test Corleone.get_bounds(cache, :rate) == (-Inf, Inf)
    @test is_parameter(cache, :input)
    @test !is_variable(cache, :input)
    @test parameter_index(cache, :input) == 1
    @test variable_index(cache, :input) === nothing
    @test variable_index(cache, :cost) == 3
    @test is_independent_variable(cache, :t)
    @test independent_variable_symbols(cache) == independent_variable_symbols(parent)
    @test constant_structure(cache) == constant_structure(parent)
    @test is_time_dependent(cache) == is_time_dependent(parent)
    if isdefined(SymbolicIndexingInterface, :is_markovian)
        @test SymbolicIndexingInterface.is_markovian(cache) == SymbolicIndexingInterface.is_markovian(parent)
    end
    @test is_timeseries_parameter(cache, :input)
    @test timeseries_parameter_index(cache, :input) == timeseries_parameter_index(parent, :input)
    @test get_all_timeseries_indexes(cache, :input) == get_all_timeseries_indexes(parent, :input)
    @test is_observed(cache, :(x + input))
    @test observed(cache, :(x + input))([2.0, 4.0, 0.0], [0.5, 1.0], 0.0) == 2.5
    @test parameter_observed(cache, :(input + rate))([0.5, 1.0], 0.0) == 1.5
    annotation = Corleone.update_annotation(Tunable(:input; kind = Val(:parameter)), cache)
    @test Corleone.get_bounds(annotation) == (0.0, 1.0)
    @test Corleone.get_default(annotation) == 0.5
    inherited = ControlSymbolCache(cache)
    @test inherited === cache
    @test inherited.sys === parent
    @test inherited.bounds === limits
    @test inherited.controls == cache.controls
    @test inherited.lagrange_variables == cache.lagrange_variables
    for kwargs in ((; controls = [:missing]), (; controls = [:x]),
            (; lagrange_variables = [:input]), (; lagrange_variables = [10]),
            (; controls = [1, :input]), (; lagrange_variables = [:cost, 3]),
            (; bounds = Dict(:missing => (0, 1))), (; bounds = Dict(1 => (0, 1))),
            (; bounds = Dict(:x => (2, 1))), (; bounds = Dict(:x => (0,))))
        @test_throws ArgumentError ControlSymbolCache(parent; kwargs...)
    end
end

@testset "Shooting metadata and annotation bounds" begin
    prob = ODEProblem((u, p, t) -> [p[1], u[1]^2], [1.0, 0.0], (0.0, 2.0), [0.5])
    limits = Dict(:p₁ => (-1.0, 2.0), :u₁ => (-2.0, 5.0))
    layer = ShootingLayer(prob, Tsit5(),
        Tunable(1; kind = Val(:parameter), lowerbound = 0.0, upperbound = 1.0) => [1.0],
        Tunable(1; kind = Val(:variable), upperbound = 3.0) => [1.0];
        controls = [1], lagrange_variables = [2], bounds = limits)
    @test symbolic_container(layer) === layer.sys
    @test control_symbols(layer.sys) == [:p₁]
    @test lagrange_symbols(layer.sys) == [:u₂]
    @test default_bounds(layer.sys) == Dict(:p₁ => (0.0, 1.0), :u₁ => (-2.0, 3.0))
    @test limits == Dict(:p₁ => (-1.0, 2.0), :u₁ => (-2.0, 5.0))
    @test Corleone.get_symbolic_container(prob) isa SymbolCache
    ps, st = LuxCore.setup(MersenneTwister(1), layer)
    trajectory, _ = layer(nothing, ps, st)
    metadata = symbolic_container(trajectory)
    @test metadata === layer.sys
    @test trajectory.sys === layer.sys
    @test control_symbols(metadata) === control_symbols(layer.sys)
    @test control_indices(metadata) == [1]
    @test lagrange_symbols(metadata) === lagrange_symbols(layer.sys)
    @test lagrange_indices(metadata) == [2]
    @test default_bounds(metadata) === default_bounds(layer.sys)
    @test Corleone.get_bounds(metadata, :p₁) == (0.0, 1.0)
    @test Corleone.get_bounds(metadata, :u₂) == (-Inf, Inf)
    for query in (control_symbols, control_indices, lagrange_symbols, lagrange_indices)
        @test_throws MethodError query(trajectory)
        @test_throws MethodError query(layer)
    end
    # Bookkeeping changes are visible through the shared cache immediately.
    default_bounds(layer.sys)[:u₂] = (0.0, Inf)
    @test Corleone.get_bounds(metadata, :u₂) == (0.0, Inf)
    @test length(first(trajectory.u)) == length(prob.u0)
    @test trajectory[:p₁] == [0.5, 0.5]
    @test trajectory[:u₂][end] > 0
    # Wrapping changes metadata, not the numerical solve or parameter layout.
    plain = ShootingLayer(prob, Tsit5(),
        Tunable(1; kind = Val(:parameter)) => [1.0],
        Tunable(1; kind = Val(:variable)) => [1.0])
    expected, _ = plain(nothing, ps, st)
    @test isempty(control_symbols(symbolic_container(expected)))
    @test isempty(lagrange_symbols(symbolic_container(expected)))
    @test isempty(default_bounds(symbolic_container(expected)))
    @test trajectory.u ≈ expected.u
    @test trajectory.residuals ≈ expected.residuals
    @test parameter_index(layer.sys, :p₁) == 1
    @test !is_variable(layer.sys, :p₁)

    cache = ControlSymbolCache(prob; controls = 1, lagrange_variables = 2, bounds = limits)
    wrapped = remake(prob; f = ODEFunction(prob.f.f; sys = cache))
    @test Corleone.get_symbolic_container(wrapped) === cache
    inherited = ShootingLayer(wrapped, Tsit5())
    @test inherited.sys.sys === cache.sys
    @test inherited.sys.bounds === limits
    @test inherited.sys.controls == cache.controls
    @test inherited.sys.lagrange_variables == cache.lagrange_variables
    # Bounds supplied by the container participate in annotation validation.
    @test_throws ArgumentError ShootingLayer(prob, Tsit5(),
        Tunable(1; kind = Val(:parameter), default = 3.0) => [1.0]; bounds = limits)
end
