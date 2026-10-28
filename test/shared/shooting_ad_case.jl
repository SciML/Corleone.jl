module ShootingADCase
using Corleone, LuxCore, OrdinaryDiffEqTsit5, Random, ComponentArrays, SciMLBase
using DiffEqBase: SensitivityADPassThrough
export backend_case, analytic_loss

scalar_dynamics(u, p, t) = [p[1] * u[1]]

function dynamics!(du, u, p, t)
    du[1] = p[1] * u[1]
    du[2] = p[1]
    return nothing
end

function backend_case(pass_through, ensemblealg)
    prob = if pass_through
        ODEProblem(
            ODEFunction{true, SciMLBase.FullSpecialize}(dynamics!), [1.0, 2.0], (0.0, 3.0), [0.2];
            sensealg = SensitivityADPassThrough(), abstol = 1.0e-11, reltol = 1.0e-11
        )
    else
        ODEProblem(dynamics!, [1.0, 2.0], (0.0, 3.0), [0.2]; abstol = 1.0e-11, reltol = 1.0e-11)
    end
    layer = ShootingLayer(
        prob, Tsit5(),
        Tunable(1; kind = Val(:variable)) => [0.0, 1.0, 2.0],
        Constant(2; kind = Val(:variable)) => [1.0, 2.0]; ensemblealg
    )
    _, st = LuxCore.setup(MersenneTwister(1), layer)
    return ShootingLoss(layer, st)
end

struct ShootingLoss{L, S}
    layer::L
    st::S
end

function (loss::ShootingLoss)(q)
    (; layer, st) = loss
    parameters = (;
        initialcondition = (;
            initialconditions = (;
                u₁ = (; lookup = (;), parameter = (; ps = q[2:4])), u₂ = (;),
            ),
        ),
        parameter = (; parameters = (; p₁ = (; lookup = (;), parameter = (; ps = q[1:1])))),
    )
    fixed = merge(
        st.initialcondition.initialconditions.u₂,
        (; parameters = (; lookup = (;), parameter = (; ps = [q[5], q[5]])))
    )
    children = merge(st.initialcondition.initialconditions, (; u₂ = fixed))
    state = merge(st, (; initialcondition = merge(st.initialcondition, (; initialconditions = children))))
    trajectory, _ = layer(nothing, ComponentArray(parameters), state)
    return sum(u -> sum(abs2, u), trajectory.u) + 0.3 * sum(abs2, shooting_constraints(trajectory))
end


# Three unit-length segments, with both sides of each reset saved. Fixed resets
# contribute samples but no continuity residuals. The global initial value is free.
function analytic_loss(q)
    a, x0, x1, x2, b = q
    e = exp(a)
    samples = (x0^2 + x1^2 + x2^2) * (1 + e^2) + 4 + (2 + a)^2 + 2 * (b^2 + (b + a)^2)
    return samples + 0.3 * ((x1 - x0 * e)^2 + (x2 - x1 * e)^2)
end


end
