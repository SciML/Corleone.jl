# ---------------------------------------------------------------------------
# DynamicFunctionLayer – a LuxCore layer wrapping arbitrary (out-of-place,
# in-place) closures over a Trajectory, for evaluating objectives/constraints.
# ---------------------------------------------------------------------------

# A real Trajectory (not a fake) to satisfy the layer's `x::Trajectory` dispatch
# and give the test closures real state data to pull from.
function _lotka_trajectory(p0, grid)
    prob = LotkaVolterra.generate()
    sol = solve(prob, Tsit5(); p = p0, saveat = grid)
    cache = Solutions.ControlSymbolCache(prob, Symbol[], [:L])
    seg = Solutions.ControlSegment(sol, cache)
    sseg = Solutions.ShootingSegment((seg,), cache)
    return Solutions.Trajectory((sseg,), cache)
end

traj = _lotka_trajectory([1.0, 1.0, 0.3], [0.0, 6.0, 12.0])

@testset "initialparameters / initialstates delegate to the stored constructors" begin
    layer = DynamicFunctionLayer(
        rng -> (; a = 1.0), rng -> (; b = 2), (x, ps, st) -> (0.0, st), nothing
    )
    ps, st = LuxCore.setup(rng, layer)
    @test ps == (; a = 1.0)
    @test st == (; b = 2)
end

@testset "out-of-place call delegates to foop and returns its (out, st)" begin
    foop = (x, ps, st) -> (sum(x[:x]) * ps.scale, merge(st, (; called = true)))
    layer = DynamicFunctionLayer(rng -> (; scale = 2.0), rng -> (; called = false), foop, nothing)
    ps, st = LuxCore.setup(rng, layer)

    out, st′ = layer(traj, ps, st)

    @test out == sum(traj[:x]) * 2.0
    @test st′.called === true
end

@testset "in-place call (fiip provided) delegates to fiip and returns (res, st)" begin
    fiip = (res, x, ps, st) -> begin
        res[1] = sum(x[:y]) * ps.scale
        merge(st, (; called = true))
    end
    layer = DynamicFunctionLayer(rng -> (; scale = 3.0), rng -> (; called = false), nothing, fiip)
    ps, st = LuxCore.setup(rng, layer)
    res = [0.0]

    res′, st′ = layer(res, traj, ps, st)

    @test res′ === res
    @test res[1] == sum(traj[:y]) * 3.0
    @test st′.called === true
end

@testset "in-place call (fiip === nothing) falls back to foop, broadcast into res" begin
    foop = (x, ps, st) -> (sum(x[:x]) - sum(x[:y]), merge(st, (; called = true)))
    layer = DynamicFunctionLayer(rng -> NamedTuple(), rng -> (; called = false), foop, nothing)
    @test layer isa DynamicFunctionLayer{<:Any, <:Any, <:Any, Nothing}
    ps, st = LuxCore.setup(rng, layer)
    res = [0.0]

    res′, st′ = layer(res, traj, ps, st)

    @test res′ === res
    @test res[1] == sum(traj[:x]) - sum(traj[:y])
    @test st′.called === true
end

@testset "in-place fallback broadcasts a vector-valued foop output elementwise" begin
    foop = (x, ps, st) -> ([sum(x[:x]), sum(x[:y])], st)
    layer = DynamicFunctionLayer(rng -> NamedTuple(), rng -> NamedTuple(), foop, nothing)
    ps, st = LuxCore.setup(rng, layer)
    res = zeros(2)

    res′, _ = layer(res, traj, ps, st)

    @test res′ === res
    @test res == [sum(traj[:x]), sum(traj[:y])]
end

@testset "DynamicFunctionLayer(f; parameters, state, isinplace) convenience constructor" begin
    @testset "defaults: empty ps/st, isinplace = nothing" begin
        foop = (x, ps, st) -> (sum(x[:x]), st)
        layer = DynamicFunctionLayer(foop)

        @test layer.foop === foop
        @test layer.fiip === nothing
        @test layer isa DynamicFunctionLayer{<:Any, <:Any, <:Any, Nothing}

        ps, st = LuxCore.setup(rng, layer)
        @test ps == (;)
        @test st == (;)

        out, _ = layer(traj, ps, st)
        @test out == sum(traj[:x])

        # in-place still works via the foop-broadcast fallback
        res = [0.0]
        res′, _ = layer(res, traj, ps, st)
        @test res′ === res
        @test res[1] == sum(traj[:x])
    end

    @testset "parameters/state/isinplace kwargs are threaded to the underlying fields" begin
        foop = (x, ps, st) -> (sum(x[:x]) * ps.scale, st)
        fiip = (res, x, ps, st) -> begin
            res[1] = sum(x[:y]) * ps.scale
            st
        end
        layer = DynamicFunctionLayer(
            foop; parameters = rng -> (; scale = 5.0), state = rng -> (; touched = false), isinplace = fiip
        )

        @test layer.foop === foop
        @test layer.fiip === fiip
        @test layer isa DynamicFunctionLayer{<:Any, <:Any, <:Any, <:Any}

        ps, st = LuxCore.setup(rng, layer)
        @test ps == (; scale = 5.0)
        @test st == (; touched = false)

        out, _ = layer(traj, ps, st)
        @test out == sum(traj[:x]) * 5.0

        res = [0.0]
        res′, _ = layer(res, traj, ps, st)
        @test res′ === res
        @test res[1] == sum(traj[:y]) * 5.0
    end
end

# ---------------------------------------------------------------------------
# process_expression – rewrites comparison expressions into (expr, lb, ub)
# ---------------------------------------------------------------------------

@testset "process_expression" begin
    @testset "bare Symbol: unconstrained" begin
        expr, lb, ub = Corleone.process_expression(:x)
        @test expr === :x
        @test lb == -Inf
        @test ub == Inf
    end

    @testset "== rewrites to (lhs - rhs, 0, 0)" begin
        # process_expression builds Expr(:call, -, args...) with the *function*
        # `-` spliced in directly (not the symbol :-), so compare against the
        # same construction rather than quoted `a - b` syntax.
        expr, lb, ub = Corleone.process_expression(:(x(3.0) - 1.0 == 0))
        @test expr == Expr(:call, -, :(x(3.0) - 1.0), 0)
        @test lb == 0.0
        @test ub == 0.0
    end

    @testset "<= rewrites to (lhs - rhs, -Inf, 0)" begin
        expr, lb, ub = Corleone.process_expression(:(x(3.0) <= 1.0))
        @test expr == Expr(:call, -, :(x(3.0)), 1.0)
        @test lb == -Inf
        @test ub == 0.0
    end

    @testset ">= rewrites to (lhs - rhs, -Inf, 0)" begin
        # Same bound pair as <=: process_expression does not flip the sense for >=.
        expr, lb, ub = Corleone.process_expression(:(x(5.0) * y(5.0) >= 3.0))
        @test expr == Expr(:call, -, :(x(5.0) * y(5.0)), 3.0)
        @test lb == -Inf
        @test ub == 0.0
    end

    @testset "other Expr (e.g. a bare objective call) passes through unconstrained" begin
        expr, lb, ub = Corleone.process_expression(:(L(12.0)))
        @test expr == :(L(12.0))
        @test lb == -Inf
        @test ub == Inf
    end

    @testset "vector of expressions processes each independently" begin
        exprs, lb, ub = Corleone.process_expression(
            [:(x(3.0) - 1.0 == 0), :(x(5.0) * y(5.0) >= 3.0)]
        )
        @test length(exprs) == 2
        @test lb == [0.0, -Inf]
        @test ub == [0.0, 0.0]
    end
end

# ---------------------------------------------------------------------------
# DynamicFunctionLayer(layer::ShootingLayer, expr...) – parses symbolic
# expressions over the shooting layer's system into (foop, fiip, state).
# ---------------------------------------------------------------------------

function _shooting_layer_fc(; n = 6)
    prob = ControlledLotka.generate()
    cgrid = collect(LinRange(0.0, 12.0, n))
    pc1 = PiecewiseParameter(:u1, copy(cgrid))
    pc2 = PiecewiseParameter(:u2, copy(cgrid))
    layer = ShootingLayer(prob, Symbol[], pc1, pc2; algorithm = Tsit5())
    return prob, layer
end

@testset "DynamicFunctionLayer(layer::ShootingLayer, expr...)" begin
    @testset "single unconstrained expression: empty ps, saveat/lb/ub in state" begin
        de_prob, shooting = _shooting_layer_fc()
        objective = DynamicFunctionLayer(shooting, :(x(3.0) + y(6.0)); rng = rng)

        ps, st = LuxCore.setup(rng, objective)
        @test ps == (;)
        @test st.lb == [-Inf]
        @test st.ub == [Inf]
        @test 3.0 in st.saveat
        @test 6.0 in st.saveat
        @test issorted(st.saveat)
        @test allunique(st.saveat)

        de = remake(de_prob, saveat = st.saveat)
        ps_shoot, st_shoot = LuxCore.setup(rng, shooting)
        traj, _ = shooting(de, ps_shoot, st_shoot)

        idx3 = findfirst(==(3.0), st.saveat)
        idx6 = findfirst(==(6.0), st.saveat)
        expected = traj[:x][idx3] + traj[:y][idx6]

        out, _ = objective(traj, ps, st)
        @test out == expected

        res = zeros(1)
        res′, _ = objective(res, traj, ps, st)
        @test res′ === res
        @test res[1] == expected
    end

    @testset "multiple comparison expressions: lb/ub match process_expression, output is a stacked vector" begin
        de_prob, shooting = _shooting_layer_fc()
        constraints = DynamicFunctionLayer(
            shooting, :(x(3.0) - 1.0 == 0), :(x(5.0) * y(5.0) >= 3.0); rng = rng
        )

        ps, st = LuxCore.setup(rng, constraints)
        @test st.lb == [0.0, -Inf]
        @test st.ub == [0.0, 0.0]
        @test 3.0 in st.saveat
        @test 5.0 in st.saveat

        de = remake(de_prob, saveat = st.saveat)
        ps_shoot, st_shoot = LuxCore.setup(rng, shooting)
        traj, _ = shooting(de, ps_shoot, st_shoot)

        idx3 = findfirst(==(3.0), st.saveat)
        idx5 = findfirst(==(5.0), st.saveat)
        expected = [traj[:x][idx3] - 1.0, traj[:x][idx5] * traj[:y][idx5] - 3.0]

        out, _ = constraints(traj, ps, st)
        @test out == expected

        res = zeros(2)
        res′, _ = constraints(res, traj, ps, st)
        @test res′ === res
        @test res == expected
    end
end
