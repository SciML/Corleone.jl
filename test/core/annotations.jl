# Load annotations in isolation so mock metadata hooks do not affect other tests.
module AnnotationTests
using Test, ConcreteStructs, SciMLBase, SymbolicIndexingInterface
include(joinpath(@__DIR__, "../../src/utilities/annotations.jl"))

# Supply model metadata through the same hooks used by symbolic extensions.
struct AnnotatedModel
    cache
end
SymbolicIndexingInterface.parameter_symbols(m::AnnotatedModel) = [:q, :p]
SymbolicIndexingInterface.variable_symbols(m::AnnotatedModel) = [:x, :y]
SymbolicIndexingInterface.parameter_index(m::AnnotatedModel, s) = SymbolicIndexingInterface.parameter_index(m.cache, s)
SymbolicIndexingInterface.variable_index(m::AnnotatedModel, s) = SymbolicIndexingInterface.variable_index(m.cache, s)
SymbolicIndexingInterface.default_values(m::AnnotatedModel) = SymbolicIndexingInterface.default_values(m.cache)
default_bounds(::AnnotatedModel) = Dict(:p => (-2, 5))
is_integer_variable(::Symbol) = true
model = AnnotatedModel(SymbolCache([:x, :y], [:p, :q], :t;
    defaults = Dict(:x => 1, :y => 2, :p => 3, :q => 4)))

@testset "Annotation construction and setters" begin
    @test_throws ArgumentError Tunable()
    @test_throws ArgumentError Constant()
    @test is_tunable(Tunable(1))
    @test !is_tunable(Constant(:p))
    @test get_index(Tunable(1)) == 1
    @test get_sym(Constant(:p)) == :p
    @test get_default(Tunable(:p => 2)) == 2
    @test get_default(Constant(1 => 3)) == 3
    @test !is_fully_specified(Tunable(1))
    @test !is_minimum_specified(Tunable(1))
    @test is_minimum_specified(set_parameter(Tunable(1)))
    @test !is_minimum_specified(Tunable(1; kind = Val(:invalid)))
    for constructor in (Tunable, Constant)
        annotation = constructor(1)
        for (setter, getter, value) in (
                (set_index, get_index, 2), (set_sym, get_sym, :p),
                (set_default, get_default, 3.0),
                (set_lowerbound, get_lowerbound, -1),
                (set_upperbound, get_upperbound, 4),
                (set_kind, get_kind, Val(:parameter)),
                (set_integer, get_integer, true),
            )
            updated = setter(annotation, value)
            @test getter(updated) == value
            @test is_tunable(updated) == is_tunable(annotation)
        end
        @test get_default(annotation) === nothing
        @test !is_integer_variable(relax_integer_variable(set_integer(annotation, true)))
    end
end

@testset "Annotation resolution" begin
    for constructor in (Tunable, Constant)
        resolved = update_annotation(set_parameter(constructor(1)), model)
        @test get_sym(resolved) == :p # enumeration deliberately starts with :q
        @test get_default(resolved) == 3
        @test get_bounds(resolved) == (-2, 5)
        @test is_integer_variable(resolved)
        @test is_fully_specified(resolved)
        @test is_tunable(resolved) == is_tunable(constructor(1))
        @test get_index(update_annotation(set_parameter(constructor(:q)), model)) == 2
        @test get_sym(update_annotation(set_variable(constructor(2)), model)) == :y
        @test get_bounds(update_annotation(set_variable(constructor(:x)), model)) == (-Inf, Inf)
        @test get_bounds(update_annotation(set_parameter(constructor(:p; lowerbound = 0)), model)) == (0, 5)
        @test get_bounds(update_annotation(set_parameter(constructor(:p; upperbound = 8)), model)) == (-2, 8)
        @test !is_integer_variable(update_annotation(set_parameter(constructor(:p; integer = false)), model))
        @test get_default(update_annotation(set_parameter(constructor(:p; default = 2)), model)) == 2
        @test get_sym(update_annotation(constructor(; index = 1, sym = :p, kind = Val(:parameter)), model)) == :p
        @test_throws ArgumentError update_annotation(constructor(1), model)
        @test_throws ArgumentError update_annotation(set_parameter(constructor(99)), model)
        @test_throws ArgumentError update_annotation(set_variable(constructor(:missing)), model)
        @test_throws ArgumentError update_annotation(constructor(; index = 1, sym = :q, kind = Val(:parameter)), model)
        @test_throws ArgumentError update_annotation(set_parameter(constructor(:p; lowerbound = 6)), model)
        @test_throws ArgumentError update_annotation(set_parameter(constructor(:p; default = 9)), model)
    end
end
end
