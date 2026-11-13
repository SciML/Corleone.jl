using Corleone: SciMLBase
using CorleoneInverse, ComponentArrays, CairoMakie, ForwardDiff, LinearAlgebra
using Optimization, OptimizationLBFGSB, OrdinaryDiffEqRosenbrock
using SymbolicIndexingInterface: SymbolCache
using TransformVariables: asℝ₊, TVScale

function egf_signaling!(du, u, p, t)
    EGFR, pEGFR, pEGFR_Akt, Akt, pAkt, S6, pAkt_S6, pS6, EGF_EGFR = u
    k₁, k₂, k₃, k₄, k₅, k₆, k₇, k₈, k₉, k₁₀, k₁₁, k₁₂, turnover, EGF = p
    binding = k₁ * EGF * EGFR - k₂ * EGF_EGFR
    akt_binding = k₃ * pEGFR * Akt - k₄ * pEGFR_Akt
    s6_binding = k₇ * pAkt * S6 - k₈ * pAkt_S6
    du[1] = -binding + turnover * (68190 - EGFR)
    du[2] = -akt_binding + k₅ * pEGFR_Akt - k₆ * pEGFR + k₁₂ * EGF_EGFR
    du[3] = akt_binding - k₅ * pEGFR_Akt
    du[4] = -akt_binding + k₁₀ * pAkt
    du[5] = k₅ * pEGFR_Akt - s6_binding + k₉ * pAkt_S6 - k₁₀ * pAkt
    du[6] = -s6_binding + k₁₁ * pS6
    du[7] = s6_binding - k₉ * pAkt_S6
    du[8] = k₉ * pAkt_S6 - k₁₁ * pS6
    du[9] = binding - k₁₂ * EGF_EGFR
    return nothing
end
states = [:EGFR, :pEGFR, :pEGFR_Akt, :Akt, :pAkt, :S6, :pAkt_S6, :pS6, :EGF_EGFR]
parameters = [:k₁, :k₂, :k₃, :k₄, :k₅, :k₆, :k₇, :k₈, :k₉, :k₁₀, :k₁₁, :k₁₂, :turnover, :EGF]
published_rates = [
    3.691e-3, 2.301e-3, 9.365e-4, 6.097e4, 0.4332, 3.016e-2, 3.273e-6,
    3.985e-4, 5.463e-6, 1.18e-2, 9.448e-4, 2.851e-2, 1.48e-3,
]
published_totals = (EGFR = 3.754e7, Akt = 2.693e-2, S6 = 188.1)
published_gains = (pAkt = 41.38, pEGFR = 5.648e-8, pS6 = 7.852e4)
u0 = zeros(9)
u0[[1, 4, 6]] .= values(published_totals)
problem = ODEProblem(
    ODEFunction(egf_signaling!; sys = SymbolCache(states, parameters, :t)),
    u0, (0.0, 3600.0), [published_rates; 0.0]; abstol = 1.0e-9, reltol = 1.0e-9
)

times = [0.0, 60.0, 120.0, 300.0, 600.0, 900.0, 1800.0, 3600.0]
doses = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
y_pAkt = [
    0.0 0.0 0.0 0.0 0.0 0.0
    0.03123 0.06221 0.3899 0.7119 0.8715 0.9973
    0.09344 0.4071 0.8231 0.8963 0.9612 0.8819
    0.2369 0.319 0.4971 0.4972 0.3735 0.2737
    0.1497 0.2059 0.2549 0.2397 0.1555 0.1118
    0.0931 0.1471 0.1938 0.1833 0.1192 0.07373
    0.06576 0.1254 0.1294 0.08957 0.06231 0.05344
    0.04577 0.06783 0.06124 0.03705 0.03678 0.03828
]
σ_pAkt = [
    0.01 0.01 0.01 0.01 0.01 0.01
    0.01 0.01 0.1 0.05 0.1 0.05
    0.01 0.03 0.07 0.05 0.025 0.02
    0.01 0.01 0.03 0.02 0.02 0.02
    0.01 0.02 0.03 0.03 0.02 0.02
    0.01 0.02 0.03 0.02 0.02 0.02
    0.01 0.02 0.02 0.02 0.02 0.02
    0.01 0.01 0.01 0.02 0.01 0.01
]
y_pEGFR = [
    0.0 0.0 0.0 0.0 0.0 0.0
    0.004955 0.0249 0.1282 0.4878 0.9095 0.9567
    0.004034 0.02299 0.1318 0.2897 0.3891 0.2689
    0.006367 0.0289 0.1212 0.2407 0.3012 0.3409
    0.01086 0.02794 0.09165 0.1385 0.1032 0.1002
    0.007135 0.03546 0.1165 0.1103 0.1062 0.0978
    0.00371 0.02123 0.05 0.04764 0.0395 0.02537
    0.006769 0.02063 0.01793 0.01783 0.0114 0.008196
]
σ_pEGFR = [
    0.01 0.01 0.01 0.01 0.01 0.01
    0.01 0.01 0.05 0.15 0.3 0.1
    0.01 0.01 0.025 0.12 0.15 0.1
    0.01 0.01 0.065 0.15 0.15 0.2
    0.01 0.01 0.025 0.065 0.05 0.05
    0.01 0.01 0.02 0.03 0.02 0.02
    0.01 0.01 0.01 0.02 0.02 0.01
    0.01 0.01 0.01 0.01 0.01 0.01
]
y_pS6 = [
    0.0 0.0 0.0 0.0 0.0 0.0
    0.001856 0.001965 0.002084 0.005719 0.007329 0.008614
    0.003342 0.002197 0.004796 0.01313 0.01442 0.01306
    0.01263 0.02012 0.04749 0.06241 0.07973 0.07993
    0.07891 0.2216 0.392 0.4115 0.4562 0.4287
    0.2552 0.5233 0.6814 0.7143 0.6529 0.6051
    0.4474 0.8585 1.0 0.9278 0.7519 0.6209
    0.2955 0.6598 0.7511 0.5506 0.3745 0.3208
]
σ_pS6 = [
    0.01 0.01 0.01 0.01 0.01 0.01
    0.01 0.01 0.01 0.01 0.01 0.01
    0.01 0.01 0.01 0.01 0.01 0.01
    0.01 0.01 0.01 0.01 0.01 0.01
    0.02 0.02 0.01 0.01 0.01 0.02
    0.02 0.03 0.025 0.035 0.02 0.03
    0.035 0.1 0.07 0.08 0.04 0.035
    0.035 0.025 0.075 0.08 0.08 0.07
]

log10_scale = asℝ₊ ∘ TVScale(log(10.0))
fitted = setdiff(1:13, [4, 9])
rates = Tunable(fitted; default = published_rates[fitted], transform = log10_scale) => nothing
fixed_rates = Constant([4, 9]) => nothing
egfr_total = Tunable(
    [1]; kind = Val(:variable), default = [published_totals.EGFR], transform = log10_scale
) => [0.0]

function dose_experiment(j)
    observations = (
        :(gain_pAkt * (pAkt + pAkt_S6)) => (times, y_pAkt[:, j], σ_pAkt[:, j]),
        :(gain_pEGFR * (pEGFR + pEGFR_Akt)) => (times, y_pEGFR[:, j], σ_pEGFR[:, j]),
        :(gain_pS6 * pS6) => (times, y_pS6[:, j], σ_pS6[:, j]),
    )
    return (; annotations = (Constant(14; default = doses[j]) => nothing,), observations)
end
names = (:egf_0_1, :egf_0_3, :egf_1, :egf_3, :egf_10, :egf_30)
experiments = Experiments(
    problem, rates, fixed_rates, egfr_total; controls = [14],
    experiments = NamedTuple{names}(map(dose_experiment, eachindex(doses)))
)

gains = map(gain -> Tunable(; default = gain, transform = log10_scale), published_gains)
shooting = ShootingLayer(problem, Rodas5P())
model = CorleoneDynamicOptProblem(
    shooting, experiments; reducer = sum,
    observation_parameters = (
        gain_pAkt = gains.pAkt, gain_pEGFR = gains.pEGFR, gain_pS6 = gains.pS6,
    )
)

q, reconstruct = pack(default_vectorizer(), model.ps)
σ = vcat(vec(σ_pAkt), vec(σ_pEGFR), vec(σ_pS6))
function negative_log_likelihood(x)
    return optimization_objective(model, reconstruct(x)) / 2 + sum(log, σ) +
        length(σ) * log(2π) / 2
end
negative_log_likelihood(q)

singular_values = svdvals(ForwardDiff.jacobian(x -> residual(model, reconstruct(x)), q))
@assert minimum(singular_values) > 1.0e-2
round.(singular_values; sigdigits = 3)

ids = reconstruct(collect(1:length(q)))
gain_columns = [ids.evaluation.gain_pAkt, ids.evaluation.gain_pEGFR, ids.evaluation.gain_pS6]
pattern = residual_jacobian_sparsity(model)
@assert vec(sum(pattern[:, gain_columns]; dims = 1)) ==
    fill(length(times) * length(doses), 3)

start = q .+ [isodd(i) ? 0.3 : -0.3 for i in eachindex(q)]
optimization = SciMLBase.remake(OptimizationProblem(model, AutoForwardDiff()); u0 = start)
fit = solve(optimization, LBFGSB(); maxiters = 1000)
@assert SciMLBase.successful_retcode(fit)
@assert negative_log_likelihood(fit.u) < negative_log_likelihood(q)
(;
    retcode = fit.retcode,
    nll_start = negative_log_likelihood(start),
    nll_published = negative_log_likelihood(q),
    nll_fit = negative_log_likelihood(fit.u),
)

estimate = ParameterEstimate(model, fit.u)

fitted_ps = reconstruct(fit.u)
trajectories, _ = shooting(experiments, fitted_ps.shooting, model.st)
fine = [0.0:0.5:10.0; 15.0:5.0:3600.0]
responses = map(trajectories) do trajectory
    p = [only(trajectory[name]) for name in parameters]
    solve(remake(problem; u0 = first(trajectory.u), p), Rodas5P(); saveat = fine)
end
fitted_gains = estimate.x[gain_columns]
signals = (
    ("pAkt", s -> fitted_gains[1] .* (s[:pAkt] .+ s[:pAkt_S6]), y_pAkt, σ_pAkt),
    ("pEGFR", s -> fitted_gains[2] .* (s[:pEGFR] .+ s[:pEGFR_Akt]), y_pEGFR, σ_pEGFR),
    ("pS6", s -> fitted_gains[3] .* s[:pS6], y_pS6, σ_pS6),
)
figure = Figure(; size = (900, 360))
for (column, (label, observe, y, noise)) in enumerate(signals)
    axis = CairoMakie.Axis(
        figure[1, column]; xlabel = "Time (s)", ylabel = label, title = label,
        xscale = Makie.pseudolog10, xticks = [0, 10, 100, 1000]
    )
    for (j, response) in enumerate(responses)
        color = Makie.wong_colors()[j]
        lines!(axis, response.t, observe(response); color, label = "$(doses[j]) ng/mL")
        errorbars!(axis, times, y[:, j], noise[:, j]; color, whiskerwidth = 4)
        scatter!(axis, times, y[:, j]; color, markersize = 6)
    end
end
Legend(figure[2, 1:3], content(figure[1, 1]); orientation = :horizontal, framevisible = false)
figure

        using Pkg
        Pkg.status()

        using InteractiveUtils
        InteractiveUtils.versioninfo()

        using Pkg
        Pkg.status(; mode = PKGMODE_MANIFEST)
