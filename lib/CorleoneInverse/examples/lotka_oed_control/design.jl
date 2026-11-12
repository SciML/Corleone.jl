using Corleone: SciMLBase
# Adapted from Christoph Plate (@chplate)'s OED-controlled estimation examples
# in https://github.com/SciML/Corleone.jl/pull/168.
# Shared by the fixed-control and multiple-shooting tutorials.
using CorleoneOED
using SymbolicIndexingInterface: parameter_index, parameter_values_at_time

function fishing_lotka!(du, u, p, t)
    du[1] = u[1] - p[2] * u[1] * u[2] - 0.4 * p[1] * u[1]
    du[2] = -u[2] + p[3] * u[1] * u[2] - 0.2 * p[1] * u[2]
    return nothing
end

function fishing_design(rng)
    problem = ODEProblem(fishing_lotka!, [0.5, 0.7], (0.0, 12.0), [0.0, 1.0, 1.0]; abstol = 1.0e-8, reltol = 1.0e-8)
    times = collect(0.0:1.0:11.0)
    switches = times[2:end]
    control = Tunable(1; default = 0.0, lowerbound = 0.0, upperbound = 1.0) => switches
    measurements = (
        ContinuousMeasurement(:w1, times, (x, p, t) -> x[1]),
        ContinuousMeasurement(:w2, times, (x, p, t) -> x[2]),
    )
    oed = OEDLayer(problem, Tsit5(), control; params = [2, 3], measurements, controls = [1])
    ps, st = LuxCore.setup(rng, oed)
    model = CorleoneDynamicOptProblem(oed, DCriterion(); ps, st, M = [4.0, 4.0])
    fit = solve(
        OptimizationProblem(model, Optimization.AutoForwardDiff()), Ipopt.Optimizer();
        print_level = 0, tol = 1.0e-6, hessian_approximation = "limited-memory", max_iter = 300
    )
    @assert SciMLBase.successful_retcode(fit)
    _, reconstruct = pack(default_vectorizer(), ps)
    trajectory, _ = oed(nothing, reconstruct(fit.u), st)
    control_index = parameter_index(trajectory, 1)
    edges = vcat(times, last(problem.tspan))
    midpoints = (edges[1:(end - 1)] .+ edges[2:end]) ./ 2
    controls = [parameter_values_at_time(trajectory, trajectory, t)[control_index] for t in midpoints]
    # Continuous measurement densities rank candidate times. Select six per species
    # for the discrete synthetic experiment, keeping each species' own schedule.
    measurement_times = map((:w1, :w2)) do name
        index = parameter_index(trajectory, name)
        weights = [parameter_values_at_time(trajectory, trajectory, t)[index] for t in times]
        return sort(times[sortperm(weights; rev = true)[1:6]])
    end
    return (; problem, switches, controls, measurement_times)
end

function fishing_observations(design, controls, rng; σ = 0.02)
    times = sort(unique(vcat(design.measurement_times...)))
    simulation = ShootingLayer(
        remake(design.problem; saveat = times), Tsit5(),
        Constant(1; default = controls) => design.switches; controls = [1]
    )
    ps, st = LuxCore.setup(rng, simulation)
    trajectory, _ = simulation(nothing, ps, st)
    data = map(enumerate(design.measurement_times)) do (i, grid)
        values = [trajectory.u[findfirst(==(t), trajectory.t)][i] for t in grid]
        return i => (grid, values .+ σ .* randn(rng, length(grid)), σ)
    end
    return Experiment(data...)
end
