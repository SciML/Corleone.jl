# Loading check for the benchmark library: every case file must define a valid game
# with its default options. No game is solved here.
using CorleoneGame, Test

const LIBRARY = normpath(joinpath(@__DIR__, "..", "..", "library"))

cases = Dict{String, Module}()
for name in sort(readdir(LIBRARY))
    path = joinpath(LIBRARY, name, name * ".jl")
    isfile(path) || continue
    modname = match(r"(?m)^module (\w+)", read(path, String))
    modname === nothing && continue
    include(path)
    cases[name] = getfield(@__MODULE__, Symbol(modname[1]))
end

@testset "Library case definitions" begin
    @test length(cases) == 25
    for (name, mod) in sort(collect(cases); by = first)
        @testset "$name" begin
            g = mod.game()
            o = mod.algorithm_options()
            CorleoneGame.validate(g, o)
            @test o.intervals <= o.refined_intervals
            @test o.intervals % o.shooting_intervals == o.refined_intervals % o.shooting_intervals == 0
            @test length(g.control_labels) == length(g.owners)
            source = read(joinpath(LIBRARY, name, name * ".jl"), String)
            @test occursin(r"^# Category: (seasonal|periodical|episodical)[,\n]", source)
            tr = simulate(g, initial_controls(g, o.intervals); samples = 2)
            @test all(isfinite, tr.objectives)
        end
    end
end

@testset "Tupelo allowance variant" begin
    include(joinpath(LIBRARY, "tupelo", "tupelo_allowance.jl"))
    g = Tupelo.game(TupeloAllowance.parameters())
    CorleoneGame.validate(g, Tupelo.algorithm_options())
    @test length(g.constraints) > length(Tupelo.game().constraints)
end
