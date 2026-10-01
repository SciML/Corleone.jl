using Corleone, CorleoneParameterEstimation
using OrdinaryDiffEqTsit5
using NonlinearSolve
using LuxCore
using Test
using Random
using StableRNGs

rng = StableRNG(11)

# Lotka-Volterra-like model with a "fishing" control slot on parameter 1. The control
# *replaces* parameter 1, so when an experiment does not prescribe fixed controls that
# parameter is held constant at its model default (`prob.p[1]`).
f = (du, u, p, t) -> (
    du[1] = u[1] - p[2] * prod(u) - 0.4 * p[1] * u[1];
    du[2] = -u[2] + p[3] * prod(u) - 0.2 * p[1] * u[2];
    nothing
)
u0 = [0.5, 0.7]
p_default = [0.0, 1.0, 1.0]   # p[1] default = 0.0 (no fishing) unless a control is applied
prob = ODEProblem(f, u0, (0.0, 12.0), p_default)

cgrid = collect(0.0:0.5:11.5)                 # control discretization (piecewise-constant)
nctrl = length(cgrid)
true_ctrl = 0.6 .+ 0.3 .* sin.(cgrid ./ 2)
obs = (u, p, t) -> u[1:2]
ts = collect(0.2:0.4:11.8)
σ = 0.05

# Generate synthetic data for a given fixed control vector on the fishing slot.
function generate_control_data(cvals)
    c1 = (1 => ControlParameter(cgrid, controls = cvals, bounds = (0.0, 1.0)),)
    lay = SingleShootingLayer(prob, Tsit5(); controls = c1, saveat = ts)
    ps, st = LuxCore.setup(rng, lay)
    ps = merge(ps, (p = [1.0, 1.0], controls = cvals))
    sol, _ = lay(nothing, ps, st)
    pe = CorleoneParameterEstimation.ParameterEstimationLayer(
        lay, ExperimentData(ts, zeros(2, length(ts))); observed = obs)
    Y = CorleoneParameterEstimation.model_values(pe, sol, ts)
    return Y .+ σ .* randn(rng, 2, length(ts))
end

# Experiment 1 is *controlled* (fishing control applied); experiment 2 is *not* (so the
# fishing parameter is held at its default p[1] = 0.0, i.e. no fishing).
function build_data(; exp1_ctrl = true_ctrl, exp2_ctrl = nothing, exp2_val = nothing)
    exps = Experiment[]
    push!(exps, Experiment(ts, generate_control_data(exp1_ctrl); σ = σ, u0 = u0,
        controls = (1 => ControlParameter(cgrid, controls = exp1_ctrl, bounds = (0.0, 1.0)),)))
    if isnothing(exp2_val)
        # experiment 2 uses its own real measurements generated with the same control
        push!(exps, Experiment(ts, generate_control_data(isnothing(exp2_ctrl) ? zeros(nctrl) : exp2_ctrl);
            σ = σ, u0 = u0, controls = isnothing(exp2_ctrl) ? nothing :
                (1 => ControlParameter(cgrid, controls = exp2_ctrl, bounds = (0.0, 1.0)),)))
    else
        push!(exps, exp2_val)
    end
    return ExperimentData(exps...)
end

@testset "Controls API" begin
    # per-experiment controls accessors
    exp1 = Experiment(ts, ones(2, length(ts)); σ = σ, u0 = u0,
        controls = (1 => ControlParameter(cgrid, controls = true_ctrl, bounds = (0.0, 1.0)),))
    exp2 = Experiment(ts, ones(2, length(ts)); σ = σ, u0 = u0)
    data = ExperimentData(exp1, exp2)
    @test has_controls(data) == true
    @test controls(data, 1) isa Tuple
    @test first(controls(data, 1)) isa Pair
    @test controls(data, 2) === nothing
    # single experiment without controls
    data_noc = ExperimentData(exp2)
    @test has_controls(data_noc) == false
    @test controls(data_noc, 1) === nothing
end

@testset "Mixed controlled / uncontrolled fit" begin
    data = build_data()
    @test n_experiments(data) == 2
    @test has_controls(data) == true
    @test controls(data, 1) !== nothing
    @test controls(data, 2) === nothing

    pe = ParameterEstimationLayer(prob, Tsit5(), data; observed = obs)
    @test n_experiments(pe.data) == 2

    # each experiment gets its own shooting layer; the uncontrolled experiment's layer
    # still carries controls so it carves out the same tunable-parameter slots, holding
    # p[1] at its default (0.0) constantly.
    ps1 = LuxCore.initialparameters(rng, pe.layer[1])
    ps2 = LuxCore.initialparameters(rng, pe.layer[2])
    @test pe.layer[1].controls !== nothing
    @test pe.layer[2].controls !== nothing
    @test ps1.controls ≈ true_ctrl atol = 1.0e-14   # controlled experiment uses its control
    @test all(iszero, ps2.controls)                 # p[1] default is 0.0

    # controls are excluded from the optimization variables: only p[2], p[3] are tuned
    inner = CorleoneParameterEstimationProblem(pe; x0 = [1.2, 1.3])
    @test length(inner.x0) == 2

    sol = solve(problem(pe; x0 = [1.2, 1.3]), GaussNewton();
        abstol = 1.0e-8, reltol = 1.0e-8, maxiters = 300)
    @test SciMLBase.successful_retcode(sol)
    res = estimate_parameters(inner, sol)
    @test isapprox(res.x, [1.0, 1.0]; atol = 0.05)
    @test res.n_parameters == 2
    @test res.n_observations == n_observations(data)
    # per-experiment fitted trajectories (one per experiment)
    fp = fitted_parameters(pe, sol)
    @test fp.trajectory isa Tuple && length(fp.trajectory) == 2
end

@testset "Controlled + uncontrolled with same non-zero control" begin
    # both experiments apply the same non-zero control; the fitter must distinguish the
    # shared tunable parameters, not fit the control.
    data = build_data(; exp1_ctrl = true_ctrl,
        exp2_val = Experiment(ts, generate_control_data(true_ctrl); σ = σ, u0 = u0,
            controls = (1 => ControlParameter(cgrid, controls = true_ctrl, bounds = (0.0, 1.0)),)))
    @test has_controls(data) == true
    @test controls(data, 1) !== nothing
    @test controls(data, 2) !== nothing
    pe = ParameterEstimationLayer(prob, Tsit5(), data; observed = obs)
    sol = solve(problem(pe; x0 = [1.2, 1.3]), GaussNewton();
        abstol = 1.0e-8, reltol = 1.0e-8, maxiters = 300)
    @test SciMLBase.successful_retcode(sol)
    res = estimate_parameters(pe, sol)
    @test isapprox(res.x, [1.0, 1.0]; atol = 0.05)
    @test length(res.x) == 2
end

@testset "Wrong control reveals an identifiable discrepancy" begin
    # Experiment 2 applies a *different* (wrong) control than the one used to generate its
    # data, so the shared tunable parameters cannot explain both experiments and the joint
    # fit is biased.
    data = build_data(; exp1_ctrl = true_ctrl,
        exp2_val = Experiment(ts, generate_control_data(true_ctrl); σ = σ, u0 = u0,
            controls = (1 => ControlParameter(cgrid, controls = 0.3 .* ones(nctrl), bounds = (0.0, 1.0)),)))
    pe = ParameterEstimationLayer(prob, Tsit5(), data; observed = obs)
    sol = solve(problem(pe; x0 = [1.2, 1.3]), GaussNewton();
        abstol = 1.0e-8, reltol = 1.0e-8, maxiters = 300)
    res = estimate_parameters(pe, sol)
    # a single shared parameter set cannot fit two experiments driven by different controls
    @test !isapprox(res.x, [1.0, 1.0]; atol = 0.15)
end

@testset "Validation" begin
    # control entries must be `index => ControlParameter` pairs
    exp_bad = Experiment(ts, ones(2, length(ts)); σ = σ, u0 = u0,
        controls = (1,))   # entry is an Int, not a Pair
    @test_throws ArgumentError ParameterEstimationLayer(prob, Tsit5(), ExperimentData(exp_bad); observed = obs)

    # control value must be a ControlParameter
    exp_bad2 = Experiment(ts, ones(2, length(ts)); σ = σ, u0 = u0, controls = (1 => [0.0, 0.1],))
    @test_throws ArgumentError ParameterEstimationLayer(prob, Tsit5(), ExperimentData(exp_bad2); observed = obs)

    # all experiments with controls must control the same parameter indices
    data_bad = ExperimentData(
        Experiment(ts, ones(2, length(ts)); σ = σ, u0 = u0,
            controls = (1 => ControlParameter(cgrid, controls = true_ctrl, bounds = (0.0, 1.0)),)),
        Experiment(ts, ones(2, length(ts)); σ = σ, u0 = u0,
            controls = (2 => ControlParameter(cgrid, controls = true_ctrl, bounds = (0.0, 1.0)),)),
    )
    @test_throws ArgumentError ParameterEstimationLayer(prob, Tsit5(), data_bad; observed = obs)

    # mismatched control discretization length is rejected
    data_bad2 = ExperimentData(
        Experiment(ts, ones(2, length(ts)); σ = σ, u0 = u0,
            controls = (1 => ControlParameter(cgrid, controls = true_ctrl, bounds = (0.0, 1.0)),)),
        Experiment(ts, ones(2, length(ts)); σ = σ, u0 = u0,
            controls = (1 => ControlParameter(cgrid[1:4], controls = ones(4), bounds = (0.0, 1.0)),)),
    )
    @test_throws ArgumentError ParameterEstimationLayer(prob, Tsit5(), data_bad2; observed = obs)
end
