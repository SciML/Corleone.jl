using Corleone: SciMLBase
using Corleone
using CorleoneOED
using OrdinaryDiffEqTsit5
using ComponentArrays
using LuxCore
using StableRNGs

using Test
using Optimization
using OptimizationMOI
using Ipopt
using LinearAlgebra
using SymbolicIndexingInterface

rng = StableRNG(1111)

include(joinpath(@__FILE__, "..", "..", "..", "..", "..", "test", "helper.jl"))

prob = LotkaVolterra.generate()

cgrid = collect(0.0:1.0:11.0)
pc1 = PiecewiseParameter(:u1, copy(cgrid), 1.0, (ps, st) -> (zeros(length(cgrid)+1), ones(length(cgrid)+1)))

tgrid1 = collect(0.0:0.5:11.5)
tgrid2 = collect(0.0:0.25:11.75)


p1 = PiecewiseParameter(:α, [0.0], 1.0, (1.0,1.0))
p2 = PiecewiseParameter(:β, [0.0], 1.0, (1.0,1.0))

w1 = ContinuousMeasurement(:w1, copy(tgrid1), (u,p,t) -> u[1:1])
w2 = ContinuousMeasurement(:w2, copy(tgrid2), (u,p,t) -> u[2:2])

oed  = OEDLayer(
    prob, [], [:α, :β], p1, p2, pc1; algorithm = Tsit5(), 
        measurements = [w1, w2],
)

ps, st = LuxCore.setup(StableRNG(1), oed)
p = ComponentArray(ps)
lb, ub = Corleone.get_bounds(oed.shooting, ps, st) .|> ComponentArray


@testset "Type stabilities and other tests" begin
    sol, _ = oed(nothing, p, st)

    @test_nowarn @inferred first(oed(nothing, p, st))
    @test sol.u[1] == vcat(prob.u0, zeros(6 + 3 ), ones(5))

    @test collect(lb.controls) == vcat(ones(2), zeros(length(cgrid) + length(tgrid1) + length(tgrid2) + 3))
    @test collect(ub.controls) == vcat(ones(2), ones(length(cgrid) + length(tgrid1) + length(tgrid2) + 3))

    res = zeros(2)
    @test_nowarn CorleoneOED.sampling_sums!(res, oed, nothing, ps, st)
    @test [12.0, 12.0] == res
    @test CorleoneOED.sampling_sums(oed, nothing, ps, st) == res[1:2]
    @test_nowarn @inferred CorleoneOED.sampling_sums(oed, nothing, ps, st)

    @test_nowarn @inferred first(CorleoneOED.fisher_information(oed, nothing, ps, st))

    fim = first(CorleoneOED.fisher_information(oed, nothing, ps, st))

    # Update starting from freshly initialized
    st_1 = CorleoneOED.update_fim(oed, [(; ps=ps)])
    @test st_1.F_init == fim
    @assert first(CorleoneOED.fisher_information(oed, nothing, ps, st_1)) == 2 * fim

    # Update starting from st_1
    st_2 = CorleoneOED.update_fim(oed, [(; ps = ps)], st_1)
    @test st_2.F_init == 2 * fim

end


@testset "Setup with discrete and continuous measurements" begin
    for w1_discrete in [true, false]
        w1 =  w1_discrete ? DiscreteMeasurement(:w1, copy(tgrid1), (u,p,t) -> u[1:1]) : ContinuousMeasurement(:w1, copy(tgrid1), (u,p,t) -> u[1:1])
        for w2_discrete in [true, false]
            w2 =  w2_discrete ? DiscreteMeasurement(:w2, copy(tgrid2), (u,p,t) -> u[2:2]) : ContinuousMeasurement(:w2, copy(tgrid2), (u,p,t) -> u[2:2])
            

            @test_nowarn oed = OEDLayer(
                prob, [], [:α, :β], p1, p2, pc1; algorithm = Tsit5(), 
                measurements = [w1, w2],
            )

            ps, st  = LuxCore.setup(rng, oed)

            @test_nowarn @inferred first(oed(nothing, ps, st))
        end
    end
end

@testset "Multiexperiments" begin
    # Case 1: Experiments for same set of parameters
    num_exp = 2
    shooting_points = [0.0, 3.0, 6.0, 9.0]
    for split in [false true]
        for shooting in [false, true]
            for w1_discrete in [true, false]
                w1 =  w1_discrete ? DiscreteMeasurement(:w1, copy(tgrid1), (u,p,t) -> u[1:1]) : ContinuousMeasurement(:w1, copy(tgrid1), (u,p,t) -> u[1:1])
                for w2_discrete in [true, false]
                    w2 =  w2_discrete ? DiscreteMeasurement(:w2, copy(tgrid2), (u,p,t) -> u[2:2]) : ContinuousMeasurement(:w2, copy(tgrid2), (u,p,t) -> u[2:2])
                    
                    measurements = [w1,w2]
                    multi = begin
                        if split
                            MultiExperimentLayer(
                                prob, Symbol[], [[2, 3], [3]], pc1;
                                algorithm = Tsit5(),
                                shooting_method = shooting ? FixedShoot(shooting_points) : NoShoot(),
                                measurements = measurements
                            )
                        else
                            MultiExperimentLayer(
                                prob, Symbol[], num_exp, pc1;
                                params = [2, 3],
                                algorithm = Tsit5(),
                                shooting_method = shooting ? FixedShoot(shooting_points) : NoShoot(),
                                measurements = measurements
                            )
                        end
                    end
                    ps, st = LuxCore.setup(StableRNG(1), multi)
                    sol, _ = multi(nothing, ps, st)

                    @test_nowarn @inferred first(multi(nothing, ps, st))
                    @test_nowarn @inferred first(CorleoneOED.fisher_information(multi, nothing, ps, st))

                    if !shooting
                        @test_nowarn @inferred CorleoneOED.sampling_sums(multi, nothing, ps, st)
                        res = zeros(2 * num_exp)
                        @test_nowarn @inferred CorleoneOED.sampling_sums!(res, multi, nothing, ps, st)

                        res_w1 = w1_discrete ? 24.0 : 12.0
                        res_w2 = w2_discrete ? 48.0 : 12.0

                        _res = begin
                            if w1_discrete && !w2_discrete
                                [res_w2, res_w1, res_w2, res_w1]
                            elseif (w2_discrete && !w1_discrete) | (w1_discrete && w2_discrete)
                                [res_w1, res_w2, res_w1, res_w2]
                            else 
                                [12.0, 12.0, 12.0, 12.0]
                            end
                        end

                        @test res == _res

                    end
                end
            end
        end
    end
end


#=
@testset "Single Experiments" begin
    multi_layer = MultipleShootingLayer(prob, Tsit5(), shooting_points..., controls = (1 => control,), bounds_p = ([1.0, 1.0], [1.0, 1.0]))

    for _layer in [layer, multi_layer]
        _oed = OEDLayer{false}(
            _layer,
            params = [2, 3],
            measurements = [
                ControlParameter(collect(tgrid1), controls = ones(length(tgrid1)), bounds = (0.0, 1.0)),
                ControlParameter(collect(tgrid2), controls = ones(length(tgrid2)), bounds = (0.0, 1.0)),
            ],
            observed = (u, p, t) -> u[1:2],
        )

        _ps, _st = LuxCore.setup(StableRNG(1), _oed)
        _p = ComponentArray(_ps)
        _optprob = OptimizationProblem(_oed, ACriterion(), M = [4.0, 4.0])

        @test _optprob.f(_optprob.u0, _optprob.p) ≈ (_layer == layer ? 0.05352869250783344 : 0.6199466255548527)

        _uopt = solve(
            _optprob, Ipopt.Optimizer(),
            tol = 1.0e-6,
            hessian_approximation = "limited-memory",
            max_iter = 100,
            print_level = 0,
        )

        @testset "Solution" begin
            @test _uopt.objective ≈ 0.03707508955468313
            @test _uopt.retcode == SciMLBase.ReturnCode.Success
            opt_p = zero(_p) .+ _uopt
            sol, _ = _oed(nothing, opt_p, _st)
            c1 = reduce(vcat, map(Base.Fix2(getindex, 10), sol.u))
            u1 = reduce(vcat, map(Base.Fix2(getindex, 11), sol.u))
            u2 = reduce(vcat, map(Base.Fix2(getindex, 12), sol.u))
            @test sol.u[1][1:2] == u0
            @test isapprox(sensitivities(_oed, sol)[end], [0.0224954223439133 -1.2367919565412857; -4.44473522371497 -2.9641449776773956], atol = 1.0e-6)
            F, _ = fisher_information(_oed, nothing, opt_p, _st)
            @test F ≈ [38.58695777364362 5.304118558316029; 5.304118558316029 92.03122059902915]
            @test sol.t[c1 .> 0.1] == [0.0, 0.15, 0.25, 0.3, 0.45, 0.5, 0.6, 0.75, 0.9, 1.0, 4.8, 4.95, 5.0, 5.1, 5.25, 5.4, 5.5, 5.55, 5.7, 5.75, 5.85, 6.0, 6.15, 6.25, 6.3, 6.45, 6.5, 6.6, 6.75, 6.9, 7.0, 7.05, 7.2, 7.25, 7.35, 7.5, 7.65, 7.75]
            @test sol.t[u1 .> 0.1] == [2.55, 2.7, 2.75, 2.85, 3.0, 3.15, 3.25, 3.3, 3.45, 3.5, 3.6, 3.75, 3.9, 4.0, 4.05, 4.2, 4.25, 4.35, 4.5, 4.65, 4.75, 4.8, 4.95, 5.0, 10.65, 10.75, 10.8, 10.95, 11.0, 11.1, 11.25, 11.4, 11.5, 11.55, 11.7, 11.75, 12.0]
            @test sol.t[u2 .> 0.1] == [3.0, 3.15, 3.25, 3.3, 3.45, 3.5, 3.6, 3.75, 3.9, 4.0, 4.05, 4.2, 4.25, 4.35, 4.5, 4.65, 4.75, 4.8, 4.95, 5.0, 5.1, 5.25, 5.4, 10.65, 10.75, 10.8, 10.95, 11.0, 11.1, 11.25, 11.4, 11.5, 11.55, 11.7, 11.75, 12.0]
        end
    end
end



=#