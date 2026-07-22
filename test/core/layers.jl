using Test

using Corleone
using Corleone: Solutions
using OrdinaryDiffEqTsit5
using StableRNGs
using LuxCore
using SymbolicIndexingInterface
using SciMLBase

include(joinpath(@__FILE__, "..", "..", "helper.jl"))

rng = StableRNG(42)

@testset "Abstract" begin
    include("layers/abstract.jl")
end

@testset "Piecewise" begin
    include("layers/piecewise_parameter.jl")
end

@testset "Controls" begin
    include("layers/controls.jl")
end

@testset "ShootingInterval" begin
    include("layers/shooting_interval.jl")
end

@testset "ShootingLayer" begin
    include("layers/shooting_layer.jl")
end
