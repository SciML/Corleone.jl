using Corleone
using LuxCore
using Random
using SciMLBase
using Test

rng = MersenneTwister(42)

@testset "ControlParameter setup and bounds" begin
    c = ControlParameter(collect(0.0:0.01:1.0))
    lb, ub = Corleone.get_bounds(c)
    ps, _ = LuxCore.setup(rng, c)

    @test ps == zero(c.t)
    @test lb == fill(-Inf, length(c.t))
    @test ub == fill(Inf, length(c.t))
    @test length(ps) == length(c.t)

    c = ControlParameter(
        collect(0.0:0.01:1.0);
        name = :test,
        controls = (rng, t) -> fill(10.0, length(t)),
        bounds = t -> (fill(-1.0, length(t)), fill(1.0, length(t))),
    )
    lb, ub = Corleone.get_bounds(c)
    ps, _ = LuxCore.setup(rng, c)

    @test all(lb .<= ps .<= ub)
    @test lb == fill(-1.0, length(ps))
    @test ub == fill(1.0, length(ps))
    @test all(ps .== 1.0)

    c = ControlParameter(
        collect(0.0:0.1:1.0);
        name = :test2,
        controls = (rng, t) -> [randn(rng, 3) for _ in eachindex(t)],
    )
    lb, ub = Corleone.get_bounds(c)
    ps, _ = LuxCore.setup(rng, c)

    @test all(lb .<= ps .<= ub)
    @test eltype(lb) == eltype(ub) == eltype(ps)
    @test length(ps) == length(c.t)

    c = ControlParameter([0.0]; name = :constant, controls = (rng, t) -> [2.5])
    ps, st = LuxCore.setup(rng, c)
    v0, st0 = @inferred c(-100.0, ps, st)
    v1, st1 = @inferred c(100.0, ps, st0)
    @test v0 == v1 == 2.5
    @test st1.current_index == 1
end


@testset "ControlParameter constructors" begin
    c_range = ControlParameter(:u => 0.0:0.5:1.0)
    @test c_range.name == :u
    @test c_range.t == collect(0.0:0.5:1.0)

    c_vec = ControlParameter(:v => [0.0, 0.5, 1.0])
    @test c_vec.name == :v
    @test c_vec.t == [0.0, 0.5, 1.0]

    c_nt = ControlParameter(
        :w => (
            t = [0.0, 1.0],
            controls = (rng, t) -> [2.0, 3.0],
            bounds = t -> (fill(-3.0, length(t)), fill(3.0, length(t))),
            shooted = true,
        ),
    )
    @test c_nt.name == :w
    @test Corleone.is_shooted(c_nt)

    @test ControlParameter(c_nt) === c_nt
    @test_throws ArgumentError ControlParameter(:not_a_valid_control)
end

@testset "ControlParameter evaluation and remake" begin
    c = ControlParameter([0.0, 0.5, 1.0]; controls = (rng, t) -> [10.0, 20.0, 30.0])
    ps, st = LuxCore.setup(rng, c)

    v, st = @inferred c(-1.0, ps, st)
    @test v == 10.0
    @test st.current_index == 1

    v, st = c(0.49, ps, st)
    @test v == 10.0
    @test st.current_index == 1

    v, st = c(0.5, ps, st)
    @test v == 20.0
    @test st.current_index == 2

    v, st = c(1.0, ps, st)
    @test v == 30.0
    @test st.current_index == 3

    c_for_remake = ControlParameter(
        [0.0, 0.5, 1.0];
        name = :u_rem,
        controls = (rng, t) -> Float64.(10 .* collect(eachindex(t))),
        bounds = t -> (fill(-100.0, length(t)), fill(100.0, length(t))),
    )

    c_same = SciMLBase.remake(c_for_remake)
    @test c_same !== c_for_remake
    @test c_same.name == c_for_remake.name
    @test c_same.t == c_for_remake.t
    @test !Corleone.is_shooted(c_same)

    c_window = SciMLBase.remake(c_for_remake; tspan = (0.25, 0.75))
    @test c_window.name == :u_rem
    @test c_window.t == [0.25, 0.5]
    @test Corleone.is_shooted(c_window)

    ps_window, st_window = LuxCore.setup(rng, c_window)
    vw0, st_window = c_window(0.1, ps_window, st_window)
    vw1, _ = c_window(0.74, ps_window, st_window)
    @test vw0 == ps_window[1]
    @test vw1 == ps_window[end]

    # Full-span window keeps all control knots.
    c_endpoint = SciMLBase.remake(c_for_remake; tspan = (0.0, 1.0))
    @test c_endpoint.t == [0.0, 0.5, 1.0]
    @test !Corleone.is_shooted(c_endpoint)

    c_empty = ControlParameter(
        Float64[];
        name = :empty,
        controls = (rng, t) -> [3.0],
        bounds = t -> (fill(-Inf, length(t)), fill(Inf, length(t))),
    )
    c_empty_remake = SciMLBase.remake(c_empty)
    @test c_empty_remake.t == Float64[]
    @test c_empty_remake.name == :empty
    @test c_empty_remake.bounds isa Function
    @test c_empty_remake.controls === c_empty.controls
end

@testset "ControlParameters container" begin
    controls = ControlParameters(
        :u => 0.0:0.5:1.0,
        :v => (
            t = [0.0, 1.0],
            controls = (rng, t) -> [7.0, 9.0],
            bounds = t -> (fill(0.0, length(t)), fill(10.0, length(t))),
        );
        transform = cs -> (sum = cs.u + cs.v, raw = cs),
    )

    ps, st = LuxCore.setup(rng, controls)
    out0, st = @inferred controls((0.25, 10.0), ps, st)
    out1, _ = @inferred controls((1.0, 10.0), ps, st)

    @test haskey(ps, :u)
    @test haskey(ps, :v)
    @test out0.p.sum == out0.p.raw.u + out0.p.raw.v
    @test out1.p.sum == out1.p.raw.u + out1.p.raw.v
    @test out0.p.raw.u == ps.u[1]
    @test out1.p.raw.u == ps.u[end]
    @test out0.p.raw.v == ps.v[1]
    @test out1.p.raw.v == ps.v[end]
end
