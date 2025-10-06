using Pkg
Pkg.activate(@__DIR__)
using Corleone
using OrdinaryDiffEq
using SciMLSensitivity
using ComponentArrays
using LuxCore
using Random

using CairoMakie
using Optimization
using OptimizationMOI
using Ipopt
using blockSQP
using LinearAlgebra

T₀ = 69 + 273.15
R = 1.987204258640
Q = 0.0131


temperatures = [40., 67., 100.] # °C
temperatures .+= 273.15 # K

# Different temperatures have different initial conditions
states_inits = [[1.7066, 8.32, 0.01, 0],
                [1.6749, 8.2262, 0.0104, 0.0017],
                [1.5608, 8.3546, 0.0082, 0.0086]]

#######################
# initial parameter estimates
#######################
p_init = [1.3978e12, 3.1760e12, 4.8880e28, 1.847e4, 1.882e4,
            2.636e4, 1.4681e-17, 6.4115e-7, 1.e-17]

function par_transform(p)
    R = 1.987204258640
    T₀ = 69 + 273.15

    k₀₁  = log(p[1] * exp(-p[4] / (R * T₀)))
    k₀₂  = log(p[2] * exp(-p[5] / (R * T₀)))
    k₀₋₁ = log(p[3] * exp(-p[6] / (R * T₀)))

    E₀₁  = p[4] / 1.e4
    E₀₂  = p[5] / 1.e4
    E₀₋₁ = p[6] / 1.e4

    lnK₁  = log(p[7])
    lnK₂  = log(p[8])
    lnK₋₁ = log(p[9])

    new_p = [k₀₁, k₀₂, k₀₋₁, E₀₁, E₀₂, E₀₋₁, lnK₁, lnK₂, lnK₋₁]
    return new_p
end


u0s = map(states_inits) do states_init
     vcat(states_init,
          0.0,
          0.0131,
          0.5 * (-p_init[8] + sqrt(p_init[8]^2 + 4 * p_init[8] * states_init[1])),
          0.5 * (-p_init[8] + sqrt(p_init[8]^2 + 4 * p_init[8] * states_init[1])),
          0.0,
          0.0)
end

# Transform parameter values
p_tranformed = par_transform(p_init)

function dow(du, u, p, t)

    y₁,y₂,y₃,y₄,y₅,y₆,y₇ ,y₈ ,y₉ ,y₁₀ = u
    dy₁,dy₂,dy₃,dy₄,dy₅,dy₆,dy₇ ,dy₈ ,dy₉ ,dy₁₀ = du
    temperature = p[10]
    k₁  = exp(p[1]) * exp(-p[4] * 1.e4/(R) * (1/temperature - 1/T₀))
    k₂  = exp(p[2]) * exp(-p[5] * 1.e4/(R) * (1/temperature - 1/T₀))
    k₋₁ = exp(p[3]) * exp(-p[6] * 1.e4/(R) * (1/temperature - 1/T₀))

    # abbreviation of ODE
    f₁ = -k₂ * y₈ * y₂
    f₂ = -k₁ * y₆ * y₂ + k₋₁ * y₁₀ - k₂ * y₈ * y₂
    f₃ = k₂ * y₈ * y₂ + k₁ * y₆ * y₄ - 0.5 * k₋₁ * y₉
    f₄ = -k₁ * y₆ * y₄ + 0.5 * k₋₁ * y₉
    f₅ = k₁ * y₆ * y₂ - k₋₁ * y₁₀
    f₆ = -k₁ * (y₆ * y₂ + y₆ * y₄) + k₋₁ * (y₁₀ + 0.5 * y₉)

    return [
       -dy₁ +  f₁;
       -dy₂ +  f₂;
       -dy₃ +  f₃;
       -dy₄ +  f₄;
       -dy₅ +  f₅;
       -dy₆ +  f₆;
        - y₇ - Q + y₆ + y₈ + y₉ + y₁₀;
        - y₈ + (exp(p[8]) * y₁) / (exp(p[8]) + y₇);
        - y₉ + (exp(p[9]) * y₃) / (exp(p[9]) + y₇);
        - y₁₀ + (exp(p[7]) * y₅) / (exp(p[7]) + y₇)
    ]
end

tspan = (0.,200.0)
du0 = zeros(10)
probs = map(zip(temperatures, u0s)) do (temperature, u0)
    _prob = DAEProblem{false}(dow, du0, u0, tspan, vcat(p_tranformed, temperature),
                #initializealg = NoInit(),
                abstol=1e-8, reltol=1e-6,
                sensealg = ForwardDiffSensitivity(),
                differential_vars = [true, true, true, true, true, true, false, false, false, false]
                )

    init_dae = init(_prob, DFBDF())

    DAEProblem{false}(dow, init_dae.du, init_dae.u, tspan, _prob.p,
                initializealg = NoInit(),
                abstol=1e-8, reltol=1e-6,
                sensealg = ForwardDiffSensitivity(),
                differential_vars = [true, true, true, true, true, true, false, false, false, false]
                )
end

sols = [solve(probs[i], DFBDF()) for i=1:3]
f = Figure()
axs = [CairoMakie.Axis(f[i,1], title="Temperature: $(temperatures[i]) K") for i=1:3]
[plot!(axs[i], sols[i]) for i=1:3]
f

observed = (u,p,t) -> u[1:4]

layer = OEDLayer(probs[1], DFBDF(); params = [4,5,6], observed = observed, dt=1.0)

ps, st = LuxCore.setup(Random.default_rng(), layer)
pc = ComponentArray(ps)
lb, ub = Corleone.get_bounds(layer)

crit = ACriterion()
ACrit = crit(layer)
ACrit(pc, nothing)

nc = length(layer.layer.controls[1].t)
dt = (-)(reverse(tspan)...)/nc

sampling_cons = let nc=nc, st = st, ax = getaxes(pc)
    (res, p, ::Any) -> begin
        ps = ComponentArray(p, ax)
        res .= [sum(ps.controls[i*nc+1:(i+1)*nc]) * dt for i=0:3]
    end
end

sampling_cons(zeros(4), collect(pc), nothing)

optfun = OptimizationFunction(
    ACrit, AutoReverseDiff(), cons = sampling_cons
)

optprob = OptimizationProblem(
    optfun, collect(pc), lb = collect(lb), ub = collect(ub), lcons=zeros(4), ucons=40.0 * ones(4)
)

uopt = solve(optprob, Ipopt.Optimizer(),
    tol = 1e-9,
    hessian_approximation = "limited-memory",
    max_iter = 300
)

optsol, _ = layer(nothing, uopt + zero(pc), st)

f = Figure()
ax = CairoMakie.Axis(f[1,1], xticks = 0:25:200, title="States")
ax2 = CairoMakie.Axis(f[1,2], xticks = 0:25:200, title="Sensitivities")
ax3 = CairoMakie.Axis(f[2,:], xticks = 0:25:200, title="Sampling")
[plot!(ax, optsol.t, sol) for sol in eachrow(Array(optsol))[1:10]]
[plot!(ax2, optsol.t, sol) for sol in eachrow(reduce(hcat, (optsol[Corleone.sensitivity_variables(layer)])))]
stairs!(ax3, 0.0:dt:last(tspan)-dt, (uopt + zero(pc)).controls[1:nc])
stairs!(ax3, 0.0:dt:last(tspan)-dt, (uopt + zero(pc)).controls[nc+1:2*nc])
stairs!(ax3, 0.0:dt:last(tspan)-dt, (uopt + zero(pc)).controls[2*nc+1:3*nc])
stairs!(ax3, 0.0:dt:last(tspan)-dt, (uopt + zero(pc)).controls[3*nc+1:4*nc])
f


oedlayers = map(probs) do prob
    OEDLayer(prob, DFBDF(); params = [4,5,6], observed = observed, dt=1.0)
end

multi_layer = Corleone.MultiExperimentLayer(oedlayers...)
#multi_layer = Corleone.MultiExperimentLayer(oedlayers[1], 3)

ps, st = LuxCore.setup(Random.default_rng(), multi_layer)
ps_multi = ComponentArray(ps)
lb_multi, ub_multi = Corleone.get_bounds(multi_layer)

crit = ACriterion()
ACrit = crit(multi_layer)
ACrit(ps_multi, st)

sampling_cons_single = let nexp = multi_layer.n_exp, nc=ACrit.nc, st = st, dt = 1.0, ax = getaxes(ps_multi)
    (res, p, ::Any) -> begin
        ps = ComponentArray(p, ax)
        samplings = reduce(vcat, map(1:nexp) do i
            local_sampling = getproperty(ps, Symbol("experiment_$i"))
            [sum(local_sampling.controls[nc[i][j]+1:nc[i][j+1]] * dt)  for j in eachindex(nc[i])[1:end-1]]
        end)
        res .= samplings
    end
end

sampling_cons_multi = let nexp = multi_layer.n_exp, nc=ACrit.nc, st = st, dt = 1.0, ax = getaxes(ps_multi)
    (res, p, ::Any) -> begin
        ps = ComponentArray(p, ax)
        samplings = reduce(vcat, map(1:nexp) do i
            local_sampling = getproperty(ps, Symbol("experiment_$i"))
            [sum(local_sampling.controls[nc[i][j]+1:nc[i][j+1]] * dt)  for j in eachindex(nc[i])[1:end-1]]
        end)
        res .= sum(samplings)
    end
end


sampling_cons_single(zeros(12), ps_multi, st)
sampling_cons_multi(zeros(1), ps_multi, st)

optfun = OptimizationFunction(
    ACrit, AutoReverseDiff(), cons = sampling_cons_multi
)

optprob = OptimizationProblem(
    optfun, collect(ps_multi), lb = collect(lb_multi), ub = collect(ub_multi), lcons=[0.0], ucons=100 * ones(1)
)

uopt = solve(optprob, Ipopt.Optimizer(),
    tol = 1e-9,
    hessian_approximation = "limited-memory",
    max_iter = 300
)


optsol, _ = multi_layer(nothing, uopt + zero(ps_multi), st)

f = Figure(size = (800, 800))
for i=1:3
    ax = CairoMakie.Axis(f[1,i], xticks = 0:25:200, title="Experiment $i\nStates")
    ax2 = CairoMakie.Axis(f[2,i], xticks = 0:25:200, title="Sensitivities")
    ax3 = CairoMakie.Axis(f[3,i], limits=(nothing, (-0.05,1.05)), xticks = 0:25:200, title="Sampling")
    [plot!(ax, optsol[i].t, sol) for sol in eachrow(Array(optsol[i]))[1:10]]
    [plot!(ax2, optsol[i].t, sol) for sol in eachrow(reduce(hcat, (optsol[i][Corleone.sensitivity_variables(multi_layer.layers[i])])))]
    local_sampling = getproperty(uopt + zero(ps_multi), Symbol("experiment_$i"))

    stairs!(ax3, 0.0:199, local_sampling.controls[1:200])
    stairs!(ax3, 0.0:199, local_sampling.controls[201:400])
    stairs!(ax3, 0.0:199, local_sampling.controls[401:600])
    stairs!(ax3, 0.0:199, local_sampling.controls[601:end])
end
f