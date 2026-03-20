using Corleone
using CorleoneOED
using OrdinaryDiffEqTsit5
using SciMLBase
using SymbolicIndexingInterface
using Test

function build_decay_problem()
    f = function (du, u, p, t)
        du[1] = -p[1] * u[1]
    end
    u0 = [1.0]
    tspan = (0.0, 2.0)
    p = [0.5]
    return ODEProblem{true}(f, u0, tspan, p)
end

function build_base_layer()
    prob = build_decay_problem()
    control = Corleone.ControlParameter(0.0:0.5:2.0, name = :u)
    return Corleone.SingleShootingLayer(prob, control; algorithm = Tsit5())
end

function solve_layer(layer; saveat = 0.0:0.1:2.0)
    prob = Corleone.get_problem(layer)
    sol = solve(prob, layer.algorithm; saveat = saveat, abstol = 1e-10, reltol = 1e-10)
    return sol
end

function analytic_solution(t, p, u0)
    x = u0 * exp(-p * t)
    dxdθ = -t * u0 * exp(-p * t)
    return x, dxdθ
end

function extract_blocks(sol, nx, np)
    trajectories = sol.u
    xs = [state[1:nx] for state in trajectories]
    Gs = [reshape(state[(nx + 1):(nx + nx * np)], nx, np) for state in trajectories]
    block_size = nx * np
    start_disc = nx + block_size + 1
    end_disc = start_disc + block_size - 1
    start_cont = end_disc + 1
    end_cont = start_cont + block_size - 1
    F_discs = [reshape(state[start_disc:end_disc], np, np) for state in trajectories]
    F_conts = [reshape(state[start_cont:end_cont], np, np) for state in trajectories]
    return xs, Gs, F_discs, F_conts
end

@testset "Augmentation v2" begin

    base_layer = build_base_layer()
    params = Symbol[:p₁]
    sens_layer = augment_sensitivities(base_layer, params)

    @testset "Sensitivities" begin
        sol = solve_layer(sens_layer; saveat = 0.0:0.2:2.0)
        for (idx, t) in enumerate(sol.t)
            x = sol.u[idx][1]
            G = sol.u[idx][2]
            x_ref, g_ref = analytic_solution(t, 0.5, 1.0)
            @test isapprox(x, x_ref; atol = 1e-9)
            @test isapprox(G, g_ref; atol = 1e-9)
        end
    end

    sys = Corleone.retrieve_symbol_cache(Corleone.get_problem(base_layer), [])
    x_symbol = SymbolicIndexingInterface.variable_symbols(sys)[1]
    expr = x_symbol
    disc_cp = Corleone.ControlParameter(:y => collect(0.0:0.5:2.0))
    cont_cp = Corleone.ControlParameter(:y_cont => collect(0.0:0.1:2.0))
    measurements = [
        DiscreteMeasurement(disc_cp) => expr,
        ContinuousMeasurement(cont_cp) => expr,
    ]
    fisher_layer = augment_fisher(sens_layer, measurements)

    @testset "Fisher Information" begin
        sol = solve_layer(fisher_layer; saveat = 0.0:0.05:2.0)
        nx = 1
        np = length(params)
        xs, Gs, Fds, Fcs = extract_blocks(sol, nx, np)
        G_values = [G[1, 1] for G in Gs]
        times = sol.t
        dt = diff(times)
        manual_disc = sum(G_values[1:end-1] .^ 2 .* dt)
        manual_cont = manual_disc
        F_disc_final = Fds[end]
        F_cont_final = Fcs[end]
        @test isapprox(F_disc_final[1, 1], manual_disc; atol = 1e-6)
        @test isapprox(F_cont_final[1, 1], manual_cont; atol = 1e-6)
    end

end
