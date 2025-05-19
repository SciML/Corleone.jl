using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using OrdinaryDiffEq
using CorleoneCore
using DifferentiationInterface, LinearAlgebra
using Zygote
using SciMLSensitivity
using Test
using Optimization
using OptimizationMOI, Ipopt
using blockSQP

@variables x(..) = 0.5 y(..) = 0.7 u(..) [input = true]
@parameters p[1:4] = [1.0; 0.4; 1.0; 0.2]  [tunable=false] #y₀ = 2.0
tspan = (0.0, 12.0)
@named first_order = ODESystem(
    [
        D(x(t)) ~ x(t) - p[1] * x(t)*y(t) -  p[2] * x(t) * u(t),
        D(y(t)) ~ -y(t) + p[3] * x(t) * y(t) - p[4] * y(t) * u(t)
    ], t, [x(t), y(t), u(t)], [p];
    costs=Num[∫((x(t) - 1)^2 + (y(t)-1)^2)],
    consolidate=sum,
    tspan=tspan,
    #initialization_eqs=[
    #    y(0) ~ y₀
    #]
)

const MAX_ITERS = 50
@testset "Permutation" begin
    N = 48
    shooting_points = [3.0, 6.0, 9.0]
    grid = ShootingGrid(shooting_points)

    controlmethod = IfElseControl(
        u(t) => (; timepoints=collect(0.0:0.25:12.0)[1:end-1], defaults=.3 * ones(N))
    )
    newsys = CorleoneCore.extend_costs(first_order)
    ctrl_sys = complete(grid(tearing(controlmethod(newsys))))

    pred = OCPredictor{false}(ctrl_sys, Tsit5(); abstol=1e-12, reltol=1e-12);

    p0 = CorleoneCore.get_p0(pred; permute=false)
    p0_perm = CorleoneCore.get_p0(pred; permute=true)

    genobj(permute::Bool) = begin
        obj = let pred=pred
            (p) -> begin
                sol = pred(p; permute=permute)
                sum(sol.mayer_variables)[1]
            end
        end
        return obj
    end

    gencons(permute::Bool) = begin
        cons = let pred=pred
            (p) -> begin
                sol= pred(p; permute=permute)
                reduce(vcat, [(-).(x...) for x in sol.shooting_variables])
            end
        end
        return cons
    end

    f, g = genobj(false), gencons(false)
    f_perm, g_perm = genobj(true), gencons(true)

    H = DifferentiationInterface.hessian(x -> f(x) + sum(g(x)), AutoZygote(), p0) |> Symmetric
    H_perm = DifferentiationInterface.hessian(x -> f_perm(x) + sum(g_perm(x)), AutoZygote(), p0_perm) |> Symmetric

    blocks_ = blockSQP.compute_hessian_blocks(H_perm)

    @test blocks_ == pred.permutation.blocks
    @test norm(H[pred.permutation.fwd,pred.permutation.fwd] .- H_perm, Inf) < 1e-2
end

@testset "Convergence" begin
    for N = 24:24:48 # Control discretization
        for shooting_points in [[6.0], [4.0, 8.0]] # Multiple Shooting lifting points

            grid = ShootingGrid(shooting_points)

            controlmethod = IfElseControl(
                u(t) => (; timepoints=LinRange(0,12,N+1)[1:end-1], defaults=.3 * ones(N))
            )

            newsys = CorleoneCore.extend_costs(first_order)
            ctrl_sys = complete(grid(tearing(controlmethod(newsys))))

            pred = OCPredictor{true}(ctrl_sys, RK4(); adaptive=false, dt=0.1);
            p0 = CorleoneCore.get_p0(pred; permute=true)

            obj = let pred=pred
                (p,x) -> begin
                    sol = pred(p; permute=true)
                    sum(sol.mayer_variables)[1]
                end
            end

            cons = let pred=pred
                (p,x_) -> begin
                    sol= pred(p; permute=true)
                    reduce(vcat, [(-).(x...) for x in sol.shooting_variables])
                end
            end

            cons_iip(res, x, p) = res .= cons(x,p)
            num_cons = length(cons(p0, []))

            lb = zeros(size(p0))
            lb[p0 .== 0.7 .|| p0 .== 0.5] .= 0.3
            ub = 5 * ones(size(p0))
            ub[p0 .== 0.3] .= 1.0

            optfun = OptimizationFunction(obj, Optimization.AutoZygote(), cons=cons_iip)
            optprob = OptimizationProblem(optfun, p0, Float64[],
                lcons = zeros(num_cons), ucons = zeros(num_cons), lb=lb, ub=ub)
            optsol_ipopt = solve(optprob, Ipopt.Optimizer();
                        tol = 1e-6, hessian_approximation="limited-memory",
                        max_iter = MAX_ITERS)
            @test SciMLBase.successful_retcode(optsol_ipopt)
            try
                optsol_bsqp = solve(optprob, BlockSQPOpt();
                        sparsity = pred.permutation.blocks,
                        options = blockSQP.sparse_options(),
                        maxiters = MAX_ITERS)
                @test SciMLBase.successful_retcode(optsol_bsqp)
            catch
                @test 0 == 1
            end
        end
    end
end
