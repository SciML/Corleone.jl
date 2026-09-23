using Test
using Corleone
using Corleone: Solutions
using Corleone.Parser: Parser, collect_leafs!, generate_grid!, replace_variables, generate_getter,
                       generate_function
using OrdinaryDiffEqTsit5
using SymbolicIndexingInterface

include(joinpath(@__FILE__, "..", "..", "helper.jl"))

_sys() = SymbolCache([:x, :y, :L], [:u1, :u2], :t)

# A real Corleone.Solutions.Trajectory over the LotkaVolterra example from
# test/helper.jl, saved exactly at `grid` so that indexing traj[:x][i] lines
# up with the i-th grid point the Parser machinery computes. `L` (the running
# cost accumulated in the 3rd state) is registered as a quadrature, matching
# how it's used elsewhere in this test file as an objective-like variable.
function _lotka_trajectory(p0, grid)
    prob = LotkaVolterra.generate()
    sol = solve(prob, Tsit5(); p = p0, saveat = grid)
    cache = Solutions.ControlSymbolCache(prob, Symbol[], [:L])
    seg = Solutions.ControlSegment(sol, cache)
    sseg = Solutions.ShootingSegment((seg,), cache)
    return Solutions.Trajectory((sseg,), cache)
end

@testset "Parser construction" begin
    sys = _sys()
    parser = Parser(sys)

    @test Set(keys(parser.variables)) == Set([:x, :y, :L])
    @test all(isempty, values(parser.variables))
    @test valtype(parser.variables) == Vector{Float64}
    @test isempty(parser.parameters)
    @test eltype(parser.parameters) == Symbol
    @test isempty(parser.indexgrid)
    @test parser.sys === sys

    parser32 = Parser{Float32}(sys)
    @test valtype(parser32.variables) == Vector{Float32}
    @test keytype(parser32.indexgrid) == Float32
end

@testset "empty!" begin
    sys = _sys()
    parser = Parser(sys)
    push!(parser.variables[:x], 1.0)
    push!(parser.parameters, :u1)
    parser.indexgrid[1.0] = 1

    empty!(parser)

    # keys survive: only the collected contents are reset, not the schema.
    @test Set(keys(parser.variables)) == Set([:x, :y, :L])
    @test all(isempty, values(parser.variables))
    @test isempty(parser.parameters)
    @test isempty(parser.indexgrid)
end

@testset "collect_leafs!" begin
    @testset "nested call collects every variable occurrence and referenced parameters" begin
        parser = Parser(_sys())
        collect_leafs!(parser, :(x(10.0) * x(22.0) + sin(y(5.0) + u1)))
        @test parser.variables[:x] == [10.0, 22.0]
        @test parser.variables[:y] == [5.0]
        @test isempty(parser.variables[:L])
        @test parser.parameters == Set([:u1])
    end

    @testset "bare parameter symbol at top level" begin
        parser = Parser(_sys())
        collect_leafs!(parser, :u2)
        @test parser.parameters == Set([:u2])
        @test all(isempty, values(parser.variables))
    end

    @testset "unrelated calls/symbols (unknown fn, independent variable) are ignored" begin
        parser = Parser(_sys())
        collect_leafs!(parser, :(foo(3.0) + t))
        @test all(isempty, values(parser.variables))
        @test isempty(parser.parameters)
    end

    @testset "repeated calls accumulate onto the same parser state" begin
        parser = Parser(_sys())
        collect_leafs!(parser, :(x(1.0)))
        collect_leafs!(parser, :(x(2.0)))
        @test parser.variables[:x] == [1.0, 2.0]
    end
end

@testset "generate_grid!" begin
    @testset "pools timepoints across all variables, dedups and sorts" begin
        parser = Parser(_sys())
        collect_leafs!(parser, :(x(10.0) * x(22.0) + sin(y(5.0) + u1)))
        collect_leafs!(parser, :(L(21.0) - x(0.0) * u2))
        generate_grid!(parser, [0.0, 12.0])

        @test parser.indexgrid ==
              Dict(0.0 => 1, 5.0 => 2, 10.0 => 3, 12.0 => 4, 21.0 => 5, 22.0 => 6)
    end

    @testset "duplicate timepoints across variables and explicit timepoints collapse to one index" begin
        parser = Parser(_sys())
        collect_leafs!(parser, :(x(5.0) + y(5.0)))
        generate_grid!(parser, [5.0, 1.0, 5.0])

        @test parser.indexgrid == Dict(1.0 => 1, 5.0 => 2)
    end
end

@testset "replace_variables" begin
    @testset "rewrites variable calls to indexed getindex, leaves everything else intact" begin
        parser = Parser(_sys())
        term = :(x(10.0) * x(22.0) + sin(y(5.0) + u1))
        collect_leafs!(parser, term)
        generate_grid!(parser, Float64[])

        replaced = replace_variables(parser, term)
        grid = parser.indexgrid
        expected = Expr(
            :call, :+,
            Expr(
                :call, :*,
                Expr(:call, getindex, :x, grid[10.0]),
                Expr(:call, getindex, :x, grid[22.0])
            ),
            Expr(:call, :sin, Expr(:call, :+, Expr(:call, getindex, :y, grid[5.0]), :u1))
        )
        @test replaced == expected
    end

    @testset "non-call term passes through unchanged" begin
        parser = Parser(_sys())
        @test replace_variables(parser, :u1) === :u1
        @test replace_variables(parser, 3.0) === 3.0
    end
end

@testset "generate_getter" begin
    @testset "only emits getters for variables actually collected, plus all parameters" begin
        parser = Parser(_sys())
        collect_leafs!(parser, :(x(3.0) + u1))
        generate_grid!(parser, Float64[])

        exprs = generate_getter(parser, :traj)
        @test length(exprs) == 2
        @test Expr(:(=), :x, Expr(:call, :getindex, :traj, QuoteNode(:x))) in exprs
        @test Expr(
            :(=), :u1,
            Expr(:call, :getindex, Expr(:call, :getproperty, :traj, QuoteNode(:ps)), QuoteNode(:u1))
        ) in exprs
    end

    @testset "no getters when nothing was collected" begin
        parser = Parser(_sys())
        generate_grid!(parser, Float64[])
        @test isempty(generate_getter(parser, :traj))
    end
end

@testset "generate_function" begin
    parser = Parser(_sys())
    term = :(x(1.0))
    collect_leafs!(parser, term)
    generate_grid!(parser, Float64[])

    oop, iip = generate_function(parser, term)

    @test oop.head == :function
    oop_call = oop.args[1]
    @test oop_call.head == :call
    @test length(oop_call.args) == 4 # fname, trajectory, ps, st

    @test iip.head == :function
    iip_call = iip.args[1]
    @test iip_call.head == :call
    @test length(iip_call.args) == 5 # fname, res, trajectory, ps, st

    iip_body = iip.args[2].args
    last_stmt = iip_body[end]
    @test last_stmt.head == :(=)
    @test last_stmt.args[1] == Expr(:ref, iip_call.args[2], 1) # res[1] = ...
end

@testset "Parser as a callable (end to end)" begin
    @testset "collects, grids, and generates one (oop, iip) pair per term" begin
        parser = Parser(_sys())
        results = parser(:(x(1.0)), :(y(2.0)); timepoints = [0.0])

        @test length(results) == 2
        @test all(pair -> pair[1].head == :function && pair[2].head == :function, results)
        @test parser.indexgrid == Dict(0.0 => 1, 1.0 => 2, 2.0 => 3)
    end

    @testset "a fresh call resets state left over from a previous call" begin
        parser = Parser(_sys())
        parser(:(x(1.0)), :(y(2.0)))
        @test parser.variables[:x] == [1.0]
        @test parser.variables[:y] == [2.0]

        parser(:(x(9.0)))
        @test parser.variables[:x] == [9.0]
        @test isempty(parser.variables[:y]) # not touched by the second call
    end

    @testset "generated functions evaluate to the correct numeric result" begin
        sys = LotkaVolterra.lotka_system # states [:x, :y, :L], params [:α, :β, :γ, :δ, :u1, :u2]
        parser = Parser(sys)
        ex1 = :(x(1.0) * x(3.0) + sin(y(2.0) + β))
        ex2 = :(L(12.0) - x(0.0))

        results = parser(ex1, ex2; timepoints = [0.0, 12.0])
        grid = parser.indexgrid
        @test grid == Dict(0.0 => 1, 1.0 => 2, 2.0 => 3, 3.0 => 4, 12.0 => 5)

        p0 = [1.0, 1.0, 0.3] # α, β, u1
        traj = _lotka_trajectory(p0, sort(collect(keys(grid))))

        # Computed independently from the same real trajectory, via ordinary
        # indexing/property access rather than the Parser-generated code path.
        expected1 = traj[:x][grid[1.0]] * traj[:x][grid[3.0]] + sin(traj[:y][grid[2.0]] + traj.ps[:β])
        expected2 = traj[:L][grid[12.0]] - traj[:x][grid[0.0]] 

        f1_oop, f1_iip = eval.(results[1])
        f2_oop, f2_iip = eval.(results[2])

        @test f1_oop(traj, nothing, nothing) == expected1
        @test f2_oop(traj, nothing, nothing) == expected2

        res1 = [0.0]
        f1_iip(res1, traj, nothing, nothing)
        @test res1[1] == expected1

        res2 = [0.0]
        f2_iip(res2, traj, nothing, nothing)
        @test res2[1] == expected2
    end
end
