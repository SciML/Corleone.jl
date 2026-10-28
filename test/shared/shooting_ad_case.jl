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


export shared_grid_case, analytic_shared_grid_loss

function shared_grid_dynamics!(du, u, p, t)
    du[1] = p[1] * u[1]
    du[2] = p[2] + p[3] * u[1]
    return nothing
end

function shared_grid_case(pass_through, ensemblealg)
    f = pass_through ? ODEFunction{true, SciMLBase.FullSpecialize}(shared_grid_dynamics!) : ODEFunction(shared_grid_dynamics!)
    kwargs = pass_through ? (; sensealg = SensitivityADPassThrough()) : (;)
    problem = ODEProblem(
        f, [1.0, 2.0], (0.0, 3.0), [0.2, 0.4, 0.1];
        abstol = 1.0e-11, reltol = 1.0e-11, kwargs...
    )
    layer = ShootingLayer(
        problem, Tsit5(),
        Tunable(1; kind = Val(:variable)) => [1.0, 2.0],
        Constant(2; kind = Val(:variable)) => [1.0, 2.0],
        Constant(3; kind = Val(:parameter)) => nothing; ensemblealg
    )
    _, st = LuxCore.setup(MersenneTwister(1), layer)
    return SharedGridLoss(layer, st)
end

struct SharedGridLoss{L, S}
    layer::L
    st::S
end

function (loss::SharedGridLoss)(q)
    layer, st = loss.layer, loss.st
    ps = (;
        initialcondition = (;
            initialconditions = (;
                u₁ = (; lookup = (;), parameter = (; u₁ = (; ps = q[3:4]), u₂ = (;))),
            ),
        ),
        parameter = (;
            parameters = (;
                p₁ = (; lookup = (;), parameter = (; p₁ = (; ps = q[1:1]), p₂ = (; ps = q[2:2]), p₃ = (;))),
            ),
        ),
    )
    fixed_u = merge(
        st.initialcondition.initialconditions.u₁.parameter.u₂,
        (; parameters = (; ps = [q[5], q[5]]))
    )
    ic_group = merge(
        st.initialcondition.initialconditions.u₁,
        (; parameter = merge(st.initialcondition.initialconditions.u₁.parameter, (; u₂ = fixed_u)))
    )
    ic_state = merge(st.initialcondition, (; initialconditions = (; u₁ = ic_group)))
    fixed_p = merge(
        st.parameter.parameters.p₁.parameter.p₃,
        (; parameters = (; ps = q[6:6]))
    )
    p_group = merge(
        st.parameter.parameters.p₁,
        (; parameter = merge(st.parameter.parameters.p₁.parameter, (; p₃ = fixed_p)))
    )
    p_state = merge(st.parameter, (; parameters = (; p₁ = p_group)))
    state = merge(st, (; initialcondition = ic_state, parameter = p_state))
    trajectory, _ = layer(nothing, ComponentArray(ps), state)
    return sum(u -> sum(abs2, u), trajectory.u) + 0.3sum(abs2, shooting_constraints(trajectory))
end

function analytic_shared_grid_loss(q)
    a, d, x1, x2, b, c = q
    e = exp(a)
    k = expm1(a) / a
    samples = sum(((1, 2), (x1, b), (x2, b))) do (x, y)
        x^2 + (x * e)^2 + y^2 + (y + d + c * x * k)^2
    end
    return samples + 0.3 * ((x1 - e)^2 + (x2 - x1 * e)^2)
end

end
