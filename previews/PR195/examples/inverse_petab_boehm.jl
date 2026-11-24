using Corleone: SciMLBase
using CorleoneInverse, CSV, DataFrames, LuxCore, Random
using OrdinaryDiffEqRosenbrock, Optimization, OptimizationLBFGSB, CairoMakie
using SymbolicIndexingInterface: parameter_values

data_dir = joinpath(pkgdir(CorleoneInverse), "examples", "petab_boehm", "data")
measurement_file = joinpath(data_dir, "measurements.tsv")
measurements = CSV.read(measurement_file, DataFrame; delim = '\t')
parameter_table = CSV.read(joinpath(data_dir, "parameters.tsv"), DataFrame; delim = '\t')
nominal = Dict(row.parameterId => row.nominalValue for row in eachrow(parameter_table))
@assert nrow(measurements) == 48
first(measurements, 5)

@assert all(ismissing, measurements.experimentId)
measurements.experimentId = fill("boehm", nrow(measurements))

measurements.noiseParameters = [10.0^nominal[id] for id in measurements.noiseParameters]

function boehm_stat5!(du, u, log_rates, t)
    A, B, AB, AA, BB, NAA, NAB, NBB = u
    degradation, export_hetero, export_homo, import_hetero, import_homo, phosphorylation = 10 .^ log_rates
    epo = 1.25e-7 * exp(-degradation * t)
    phos_AA = epo * phosphorylation * A^2
    phos_AB = epo * phosphorylation * A * B
    phos_BB = epo * phosphorylation * B^2
    imp_AA, imp_AB, imp_BB = import_homo * AA, import_hetero * AB, import_homo * BB
    exp_AA, exp_AB, exp_BB = export_homo * NAA, export_hetero * NAB, export_homo * NBB
    cyt_to_nuc, nuc_to_cyt = 1.4 / 0.45, 0.45 / 1.4
    du[1] = -2phos_AA - phos_AB + nuc_to_cyt * (2exp_AA + exp_AB)
    du[2] = -2phos_BB - phos_AB + nuc_to_cyt * (2exp_BB + exp_AB)
    du[3] = phos_AB - imp_AB
    du[4] = phos_AA - imp_AA
    du[5] = phos_BB - imp_BB
    du[6] = cyt_to_nuc * imp_AA - exp_AA
    du[7] = cyt_to_nuc * imp_AB - exp_AB
    du[8] = cyt_to_nuc * imp_BB - exp_BB
    return nothing
end

specificity = nominal["specC17"]
function pSTAT5A(u, p, t)
    A, B, AB, AA, BB = u
    return (200AA * specificity + 100AB) / (A * specificity + 2AA * specificity + AB)
end
function pSTAT5B(u, p, t)
    A, B, AB, AA, BB = u
    return (-100AB + 200BB * (specificity - 1)) / (B * (specificity - 1) - AB + 2BB * (specificity - 1))
end
function rSTAT5A(u, p, t)
    A, B, AB, AA, BB = u
    return (100A * specificity + 200AA * specificity + 100AB) /
        (A * specificity - B * (specificity - 1) + 2AA * specificity + 2AB - 2BB * (specificity - 1))
end
observable_maps = (
    pSTAT5A_rel = pSTAT5A, pSTAT5B_rel = pSTAT5B, rSTAT5A_rel = rSTAT5A,
)

u0 = [207.6 * nominal["ratio"], 207.6 * (1 - nominal["ratio"]), zeros(6)...]
data = experiments_from_dataframe(
    measurements;
    observables = observable_maps, initial_states = (boehm = u0,)
)
@assert n_observations(data) == nrow(measurements)

rate_ids = [
    "log10_Epo_degradation_BaF3", "log10_k_exp_hetero", "log10_k_exp_homo",
    "log10_k_imp_hetero", "log10_k_imp_homo", "log10_k_phos",
]
published_log_rates = [nominal[id] for id in rate_ids]

starting_log_rates = published_log_rates .+ [0.2, 0.4, -0.15, 0.15, -0.3, -0.15]
bounds = Dict(row.parameterId => (row.lowerBound, row.upperBound) for row in eachrow(parameter_table))
lower = [bounds[id][1] for id in rate_ids]
upper = [bounds[id][2] for id in rate_ids]
problem = ODEProblem(
    boehm_stat5!, u0, extrema(measurements.time), starting_log_rates;
    abstol = 1.0e-9, reltol = 1.0e-9
)

shooting = ShootingLayer(
    problem, Rodas5P(),
    Tunable(1:6; lowerbound = lower, upperbound = upper) => nothing
)
experiments = Experiments(shooting, data)
ps, st = LuxCore.setup(Xoshiro(42), experiments)
model = CorleoneDynamicOptProblem(shooting, experiments; ps, st)
optimization = OptimizationProblem(model, AutoForwardDiff())
_, reconstruct = pack(default_vectorizer(), model.ps)
initial_objective = optimization_objective(model)
fit = solve(optimization, LBFGSB(); maxiters = 1000)
@assert SciMLBase.successful_retcode(fit)
fitted_objective = optimization_objective(model, reconstruct(fit.u))
@assert fitted_objective < initial_objective / 2
summary = (retcode = fit.retcode, initial_objective, fitted_objective, measurements = n_observations(data))
@show summary

fitted_ps = reconstruct(fit.u)
fitted_trajectories, _ = shooting(experiments, fitted_ps, model.st)

fitted_log_rates = parameter_values(first(fitted_trajectories))
rate_comparison = DataFrame(
    parameterId = rate_ids,
    startingRate = 10 .^ starting_log_rates, publishedRate = 10 .^ published_log_rates,
    fittedRate = 10 .^ fitted_log_rates
)
@show rate_comparison

plot_state = merge(model.st, (; saveat = sort!(unique!(vcat(collect(0.0:0.5:240.0), measurements.time)))))
trajectories, _ = shooting(experiments, fitted_ps, plot_state)
predictions = trajectories_dataframe((boehm = only(trajectories),); observables = observable_maps)

figure = Figure(; size = (1000, 340))
for (i, id) in enumerate(keys(observable_maps))
    axis = CairoMakie.Axis(figure[1, i]; title = string(id), xlabel = "Time (min)", ylabel = "Relative abundance (%)")
    observed = measurements[measurements.observableId .== string(id), :]
    predicted = predictions[predictions.observableId .== string(id), :]
    errorbars!(axis, observed.time, observed.measurement, observed.noiseParameters; color = :gray)
    scatter!(axis, observed.time, observed.measurement; label = "Measurements", markersize = 6)
    lines!(axis, predicted.time, predicted.simulation; label = "Fitted model", color = :orange)
    i == 1 && axislegend(axis; position = :rt)
end
figure

        using Pkg
        Pkg.status()

        using InteractiveUtils
        InteractiveUtils.versioninfo()

        using Pkg
        Pkg.status(; mode = PKGMODE_MANIFEST)
