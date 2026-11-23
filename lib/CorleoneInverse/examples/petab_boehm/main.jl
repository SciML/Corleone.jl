# ---
# title: Fit published PEtab measurements with DataFrames
# description: Read the Boehm STAT5 dataset, fit six kinetic rates, and export observable predictions.
# category: parameter_estimation
# id: inverse_petab_boehm
# julia: "1.10"
# notebook: false
# generate_cover: false
# ---

# [Boehm et al. (2014)](https://doi.org/10.1021/pr5006923) measured the
# phosphorylation and dimerization of STAT5A and STAT5B after erythropoietin (Epo)
# stimulation. We fit their eight-state reaction model to 48 measurements of
# three relative observables, sampled from 0 to 240 minutes.
#
# The input files are the unmodified PEtab 2 tables from the
# [PEtab benchmark collection](https://github.com/PEtab-dev/petab-benchmark-problems/tree/49900951a5594a757ad5e651684c7f0a59272881/problems/Boehm_JProteomeRes2014/v2).
# Copies at that fixed revision are bundled with this script, so running the
# tutorial needs no network connection. See `data/README.md` for provenance.
#
# !!! info "What you will learn"
#     Read real PEtab measurements into a DataFrame, resolve noise references,
#     map observable IDs to Julia callbacks, fit a stiff model, and export the
#     fitted observables with the same IDs.
using SciMLBase: successful_retcode
using CorleoneInverse, CSV, DataFrames, LuxCore, Random
using OrdinaryDiffEqRosenbrock, Optimization, OptimizationLBFGSB, CairoMakie
using SymbolicIndexingInterface: parameter_values

# ## Read the measurement and parameter tables
# `pkgdir` also locates the bundled files when this script runs from the generated
# documentation page. To use your own PEtab file, replace `measurement_file` with
# its path. CSV.jl reads the tab-separated file; the Corleone extension handles
# the DataFrame after the explicit preparation below.
data_dir = joinpath(pkgdir(CorleoneInverse), "examples", "petab_boehm", "data")
measurement_file = joinpath(data_dir, "measurements.tsv")
measurements = CSV.read(measurement_file, DataFrame; delim = '\t')
parameter_table = CSV.read(joinpath(data_dir, "parameters.tsv"), DataFrame; delim = '\t')
nominal = Dict(row.parameterId => row.nominalValue for row in eachrow(parameter_table))
@assert nrow(measurements) == 48
first(measurements, 5)

# Empty `experimentId` cells mean "use the model as is" in PEtab 2. This dataset
# has one such experiment, which we name `boehm` for Corleone's named container.
# The original measurement file stays unchanged.
@assert all(ismissing, measurements.experimentId)
measurements.experimentId = fill("boehm", nrow(measurements))

# `noiseParameters` contains parameter IDs, rather than standard deviations.
# In this dataset `observables.tsv` defines normally distributed noise with
# `noiseFormula = 10 ^ noiseParameter1_<observableId>`. Resolve those references
# explicitly before import. We hold these three published noise values fixed:
# this is a conditional weighted least-squares fit of six kinetic rates, rather
# than the full PEtab fit of six rates plus three noise parameters.
measurements.noiseParameters = [10.0^nominal[id] for id in measurements.noiseParameters]

# ## Transcribe the published reaction model
# The state order matches the source SBML: cytosolic STAT5A, STAT5B, pApB,
# pApA, pBpB, followed by nuclear pApA, pApB, and pBpB. Homodimer formation
# consumes two monomers. Nuclear export returns dephosphorylated monomers.
# Compartment volumes are `cyt = 1.4` and `nuc = 0.45`; transport therefore
# needs volume factors when expressed as concentration derivatives.
#
# Parameters are the six base-10 logarithms used in `parameters.tsv`. The
# time-dependent Epo concentration is `1.25e-7 * exp(-Epo_degradation_BaF3*t)`.
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

# These observation formulas reproduce `observables.tsv`, with its fixed
# factor `specC17`. They return percentages, matching the data's
# scale. PEtab formulas are written out as Julia functions; the adapter does
# not parse SBML or evaluate arbitrary PEtab formulas.
specificity = nominal["specC17"]
function pSTAT5A(u, p, t)
    A, B, AB, AA, BB, _, _, _ = u
    return (200AA * specificity + 100AB) / (A * specificity + 2AA * specificity + AB)
end
function pSTAT5B(u, p, t)
    A, B, AB, AA, BB, _, _, _ = u
    return (-100AB + 200BB * (specificity - 1)) / (B * (specificity - 1) - AB + 2BB * (specificity - 1))
end
function rSTAT5A(u, p, t)
    A, B, AB, AA, BB, _, _, _ = u
    return (100A * specificity + 200AA * specificity + 100AB) /
        (A * specificity - B * (specificity - 1) + 2AA * specificity + 2AB - 2BB * (specificity - 1))
end
observable_maps = (
    pSTAT5A_rel = pSTAT5A, pSTAT5B_rel = pSTAT5B, rSTAT5A_rel = rSTAT5A,
)

# Check the three maps against independently evaluated observables.tsv formulas
# at an explicit eight-state probe, before asking an optimizer to fit anything.
# These are formula reference values, not additional experimental measurements.
reference_state = [90.0, 65.0, 12.0, 8.0, 6.0, 5.0, 7.0, 9.0]
reference_observations = (
    pSTAT5A_rel = 58.743895124667979,
    pSTAT5B_rel = 28.127437748418171,
    rSTAT5A_rel = 22.422024341277389,
)
for id in keys(observable_maps)
    @assert isapprox(observable_maps[id](reference_state, nothing, 0.0), reference_observations[id]; rtol = 1.0e-12)
end

# ## Import the DataFrame and fit the kinetic rates
# No measurements are generated or averaged. The named tuple returned by the
# importer is ready to pass to `Experiments`; additional experiment IDs would
# produce additional entries automatically.
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
# Use round starting guesses chosen independently of the published optimum.
starting_log_rates = [-2.0, -2.0, -2.0, -2.0, 2.0, 3.0]
bounds = Dict(row.parameterId => (row.lowerBound, row.upperBound) for row in eachrow(parameter_table))
lower = [bounds[id][1] for id in rate_ids]
upper = [bounds[id][2] for id in rate_ids]
problem = ODEProblem(
    boehm_stat5!, u0, extrema(measurements.time), starting_log_rates;
    abstol = 1.0e-9, reltol = 1.0e-9
)
# Fast homodimer import makes this system stiff, so use an implicit solver.
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
@assert successful_retcode(fit)
fitted_objective = optimization_objective(model, reconstruct(fit.u))
@assert fitted_objective < initial_objective / 2
@assert fitted_objective < 1.1 # Reference quality: published noise-scaled MSE is approximately one.
summary = (retcode = fit.retcode, initial_objective, fitted_objective, measurements = n_observations(data))
@show summary

# The default objective is the mean squared, noise-scaled residual. In this run
# it falls from about 37.85 to 1.00 from the independent starting guess. A value
# near one means the remaining discrepancies are on the scale of the fixed
# measurement standard deviations.
#
# Exponentiate the optimizer's log10 coordinates to report physical rates. Some
# fitted rates reach the published bounds. This example reports one local
# solution; parameter uncertainty requires additional analysis.
fitted_ps = reconstruct(fit.u)
fitted_trajectories, _ = shooting(experiments, fitted_ps, model.st)
# Read the complete rate vector from the trajectory's parameter interface.
fitted_log_rates = parameter_values(first(fitted_trajectories))
rate_comparison = DataFrame(
    parameterId = rate_ids,
    startingRate = 10 .^ starting_log_rates, publishedRate = 10 .^ published_log_rates,
    fittedRate = 10 .^ fitted_log_rates
)
@show rate_comparison

# ## Export predictions and compare them with the data
# Save a dense grid for plotting, including every measurement time. Use the
# original observable IDs on export, so measurement and simulation tables can
# be joined on `(observableId, experimentId, time)`.
plot_state = merge(model.st, (; saveat = sort!(unique!(vcat(collect(0.0:0.5:240.0), measurements.time)))))
trajectories, _ = shooting(experiments, fitted_ps, plot_state)
predictions = trajectories_dataframe((boehm = only(trajectories),); observables = observable_maps)
# To save the PEtab-style simulation table:
# `CSV.write("boehm_simulations.tsv", select(predictions,
#     :observableId, :experimentId, :time, :simulation); delim = '\t')`.

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
