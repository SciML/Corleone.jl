using CorleoneOED, OrdinaryDiffEqTsit5, ComponentArrays, LuxCore, StableRNGs
using Test, Optimization, OptimizationMOI, Ipopt, LinearAlgebra
using SymbolicIndexingInterface: parameter_values_at_time, parameter_index
function lotka_dynamics(u, p, t)
    return [
        u[1] - p[2] * prod(u[1:2]) - 0.4 * p[1] * u[1];
        -u[2] + p[3] * prod(u[1:2]) - 0.2 * p[1] * u[2]
    ]
end

tspan = (0.0, 12.0)
u0 = [0.5, 0.7]
p0 = [0.0, 1.0, 1.0]

prob = ODEProblem{false}(
    lotka_dynamics, u0, tspan, p0,
)


cgrid = 0.0:0.25:11.75
tgrid1 = 0.0:0.5:11.5
tgrid2 = 0.0:0.15:11.75
shooting_points = [3.0, 6.0, 9.0]
function lotka_layer(;
        discrete = false, fixed = false, shooting = false, params = [2, 3], svd = false,
        grids = (tgrid1, tgrid2)
    )
    control = Tunable(1; default = 0.0, lowerbound = 0.0, upperbound = 1.0) => collect(cgrid[2:end])
    resets = Tunable([1, 2]; kind = Val(:variable), default = u0) => shooting_points
    definitions = ((fixed ? () : (control,))..., (shooting ? (resets,) : ())...)
    measurement = discrete ? DiscreteMeasurement : ContinuousMeasurement
    measurements = (
        measurement(:w1, collect(grids[1]), (u, p, t) -> u[1]),
        measurement(:w2, collect(grids[2]), (u, p, t) -> u[2]),
    )
    return OEDLayer(prob, Tsit5(), definitions...; params, measurements, svd)
end
# Legacy regression schedules describe the incoming interval at each saved time.
# Query its left limit; production histories remain right-continuous.
parameter_samples(traj, id) = [parameter_values_at_time(traj, traj, t == first(traj.t) ? t : prevfloat(t))[parameter_index(traj, id)] for t in traj.t]

# Independent numerical variational equations validate the generated augmentation
# at the current design, rather than freezing an optimizer's last few digits.
function reference_lotka_information(oed, trajectory)
    times = sort!(unique!(vcat(collect(prob.tspan), trajectory.st.timestops)))
    value = vcat(u0, zeros(4 + 4))
    for (left, right) in zip(times[1:(end - 1)], times[2:end])
        physical = parameter_values_at_time(trajectory, trajectory, left)
        c, β, γ = physical[1:3]
        w1 = physical[parameter_index(trajectory, :w1)]
        w2 = physical[parameter_index(trajectory, :w2)]
        function dynamics(u, p, t)
            x, y = u[1:2]
            G = reshape(u[3:6], 2, 2)
            fx = [1 - β * y - 0.4 * c -β * x; γ * y -1 + γ * x - 0.2 * c]
            fp = [-x * y 0.0; 0.0 x * y]
            dG = fx * G + fp
            dF = G' * Diagonal([w1, w2]) * G
            return vcat(lotka_dynamics(u[1:2], [c, β, γ], t), vec(dG), vec(dF))
        end
        solution = solve(
            ODEProblem(dynamics, value, (left, right)), Tsit5();
            abstol = 1.0e-12, reltol = 1.0e-12, save_everystep = false
        )
        value = last(solution.u)
    end
    return reshape(value[3:6], 2, 2), reshape(value[7:10], 2, 2)
end

function accurate_prediction(oed, ps, st)
    problem = remake(oed.augmented_prob; abstol = 1.0e-12, reltol = 1.0e-12)
    base = oed.shooting
    layer = ShootingLayer(
        base.name, problem, base.alg, base.ensemblealg,
        base.initialcondition, base.parameter, base.sys
    )
    initialized = deepcopy(ps)
    forward_initialization(StableRNG(1), layer, initialized, st)
    return first(oed(problem, initialized, st))
end
