module LotkaVolterra
    using SymbolicIndexingInterface
    using SciMLBase

    const lotka_system = SymbolCache(
        [:x, :y, :L], [:α, :β, :γ, :δ, :u1, :u2,], :t
    )

    function generate(; sys=lotka_system, kwargs...)

        function lotka_dynamics!(du, u, p, t)
            du[1] = p[1]*u[1] - p[2] * prod(u[1:2]) - 0.4 * p[5] * u[1]
            du[2] = -p[4]*u[2] + p[3] * prod(u[1:2]) - 0.2 * p[6] * u[2]
            return du[3] = (u[1] - 1.0)^2 + (u[2] - 1.0)^2
        end

        tspan = (0.0, 12.0)
        u0 = [0.5, 0.7, 0.0]
        p0 = [1., 1., 1., 1., 1., 1.]

        prob = ODEProblem(ODEFunction(lotka_dynamics!, sys=sys), u0, tspan, p0; abstol=1.0e-8, reltol=1.0e-6)
    end
end