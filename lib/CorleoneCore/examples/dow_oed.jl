using Pkg
Pkg.activate(joinpath(@__DIR__, "../"))
using CorleoneCore
using OrdinaryDiffEq
using SciMLSensitivity
using ComponentArrays
using LuxCore
using Random

using CairoMakie
using BenchmarkTools
using Zygote
using ForwardDiff

using Optimization
using OptimizationMOI
using Ipopt
using blockSQP
using LinearAlgebra

temperature = 40
T₀ = 69 + 273.15
R = 1.987204258640
Q = 0.0131

using OptimizationMOI.ModelingToolkit
#######################
# given starting values for the first four states
#######################
if temperature == 40
    states_init = [1.7066, 8.32, 0.01, 0]
elseif temperature == 67
    states_init = [1.6749, 8.2262, 0.0104, 0.0017]
else
    states_init = [1.5608, 8.3546, 0.0082, 0.0086]
end

# rescale to Kelvin
temperature += 273.15

#######################
# initial parameter estimates
#######################

p_init = [1.3978e12,3.1760e12,4.8880e28,1.847e4,1.882e4,2.636e4,1.4681e-17,6.4115e-7,1.e-17]

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

u0 = vcat(states_init,
          0.0,
          0.0131,
          0.5 * (-p_init[8] + sqrt(p_init[8]^2 + 4 * p_init[8] *states_init[1])),
          0.5 * (-p_init[8] + sqrt(p_init[8]^2 + 4 * p_init[8] * states_init[1])),
          0.0,
          0.0)

# select for each parameter whether it shall be estimated or fixed
p_tranformed = par_transform(p_init)
estimation_choice = [1, 0, 0, 1, 0, 0, 0, 0, 0]
p_fix_init = []
p_tune_init = []
for i in eachindex(estimation_choice)
    global p_tranformed, p_fix_init, p_tune_init
    curr_par = p_tranformed[i]
    if estimation_choice[i] == 1
        p_tune_init = vcat(p_tune_init, curr_par)
    else
        p_fix_init = vcat(p_fix_init, curr_par)
    end
end

tunable_start = false

@variables t
if tunable_start
    @variables y₁(t) = states_init[1] [description = "Variable 1", tunable = true, bounds = (states_init[1] * 0.5, states_init[1] * 2)]
    @variables y₂(t) = states_init[2] [description = "Variable 2", tunable = true, bounds = (states_init[2] * 0.5, states_init[2] * 2)]
    @variables y₃(t) = states_init[3] [description = "Variable 3", tunable = true, bounds = (states_init[3] * 0.5, states_init[3] * 2)]
    @variables y₄(t) = states_init[4] [description = "Variable 4", tunable = true, bounds = (states_init[4] * 0.5, states_init[4] * 2)]
else
    @variables y₁(t) = states_init[1] [description = "Variable 1"]
    @variables y₂(t) = states_init[2] [description = "Variable 2"]
    @variables y₃(t) = states_init[3] [description = "Variable 3"]
    @variables y₄(t) = states_init[4] [description = "Variable 4"]
end

@variables y₅(t) = 0. [description = "Variable 5"]
@variables y₆(t) = 0.0131 [description = "Variable 6"]
@variables y₇(t) = 0.5 * (-p_init[8] + sqrt(p_init[8]^2 + 4*p_init[8]*states_init[1])) [description = "Variable 7"]
@variables y₈(t) = 0.5 * (-p_init[8] + sqrt(p_init[8]^2 + 4*p_init[8]*states_init[1])) [description = "Variable 8"]
@variables y₉(t) = 0. [description = "Variable 9"]
@variables y₁₀(t) = 0. [description = "Variable 10"]

@variables u(t) [description = "Control"]
@parameters p_fix[1:length(p_fix_init)] = p_fix_init [description = "Fixed Parameters", tunable = false]
@parameters p_tune[1:length(p_tune_init)] = p_tune_init [description = "Tunable Parameters", tunable = true]
D = Differential(t)
#@variables obs(t)[1:4] [description = "Observed", measurement_rate = 50]
#obs = collect(obs)

# summarize all parameters
p_est = []
tune_counter, fix_counter = 1, 1
for i in eachindex(estimation_choice)
    global p_fix, p_tune, p_est
    global tune_counter, fix_counter
    if estimation_choice[i] == 1
        p_est = vcat(p_est, p_tune[tune_counter])
        tune_counter += 1
    else
        p_est = vcat(p_est, p_fix[fix_counter])
        fix_counter += 1
    end
end

p_est

print("p_est: ", p_est)

# functions of tunable parameters
k₁  = exp(p_est[1]) * exp(-p_est[4] * 1.e4/(R) * (1/temperature - 1/T₀))
k₂  = exp(p_est[2]) * exp(-p_est[5] * 1.e4/(R) * (1/temperature - 1/T₀))
k₋₁ = exp(p_est[3]) * exp(-p_est[6] * 1.e4/(R) * (1/temperature - 1/T₀))

# abbreviation of ODE
f₁ = -k₂ * y₈ * y₂
f₂ = -k₁ * y₆ * y₂ + k₋₁ * y₁₀ - k₂ * y₈ * y₂
f₃ = k₂ * y₈ * y₂ + k₁ * y₆ * y₄ - 0.5 * k₋₁ * y₉
f₄ = -k₁ * y₆ * y₄ + 0.5 * k₋₁ * y₉
f₅ = k₁ * y₆ * y₂ - k₋₁ * y₁₀
f₆ = -k₁ * (y₆ * y₂ + y₆ * y₄) + k₋₁ * (y₁₀ + 0.5 * y₉)

@named dow_oed = System(
    [
        D(y₁) ~  f₁;
        D(y₂) ~  f₂;
        D(y₃) ~  f₃;
        D(y₄) ~  f₄;
        D(y₅) ~  f₅;
        D(y₆) ~  f₆;
        0 ~ - y₇ - Q + y₆ + y₈ + y₉ + y₁₀;
        0 ~ - y₈ + (exp(p_est[8]) * y₁) / (exp(p_est[8]) + y₇);
        0 ~ - y₉ + (exp(p_est[9]) * y₃) / (exp(p_est[9]) + y₇);
        0 ~ - y₁₀ + (exp(p_est[7]) * y₅) / (exp(p_est[7]) + y₇)
    ], t, [y₁,y₂,y₃,y₄,y₅,y₆,y₇ ,y₈ ,y₉ ,y₁₀], vcat(p_tune, p_fix)
)


p = Symbolics.getdefaultval.(p_est)
u0

function dow(du, u, p_est, t)

    y₁,y₂,y₃,y₄,y₅,y₆,y₇ ,y₈ ,y₉ ,y₁₀ = u
    dy₁,dy₂,dy₃,dy₄,dy₅,dy₆,dy₇ ,dy₈ ,dy₉ ,dy₁₀ = du
    k₁  = exp(p_est[1]) * exp(-p_est[4] * 1.e4/(R) * (1/temperature - 1/T₀))
    k₂  = exp(p_est[2]) * exp(-p_est[5] * 1.e4/(R) * (1/temperature - 1/T₀))
    k₋₁ = exp(p_est[3]) * exp(-p_est[6] * 1.e4/(R) * (1/temperature - 1/T₀))

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
        - y₈ + (exp(p_est[8]) * y₁) / (exp(p_est[8]) + y₇);
        - y₉ + (exp(p_est[9]) * y₃) / (exp(p_est[9]) + y₇);
        - y₁₀ + (exp(p_est[7]) * y₅) / (exp(p_est[7]) + y₇)
    ]
end

tspan = (0.,200.0)
du0 = zeros(10)
prob = DAEProblem{false}(dow,du0,u0, tspan, p,
            #initializealg = NoInit(),
            abstol=1e-8, reltol=1e-6,
            sensealg = ForwardDiffSensitivity(),
            differential_vars = [true, true, true, true, true, true, false, false, false, false]
            )

init_dae = init(prob, DFBDF())

prob = DAEProblem{false}(dow,init_dae.du, init_dae.u, tspan, p,
            initializealg = NoInit(),
            abstol=1e-8, reltol=1e-6,
            sensealg = ForwardDiffSensitivity(),
            differential_vars = [true, true, true, true, true, true, false, false, false, false]
            )

sol = solve(prob, DFBDF())
du0 = sol(0.0, Val{1})
plot(sol)

observed = (u,p,t) -> u[1:4]

layer = OEDLayer(prob, DFBDF(); params = [4,5,6], observed = observed, dt=1.0)

ps, st = LuxCore.setup(Random.default_rng(), layer)
pc = ComponentArray(ps)
lb, ub = CorleoneCore.get_bounds(layer)

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

@elapsed begin
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
end
optsol, _ = layer(nothing, uopt + zero(pc), st)

f = Figure()
ax = CairoMakie.Axis(f[1,1], xticks = 0:25:200, title="States")
ax2 = CairoMakie.Axis(f[1,2], xticks = 0:25:200, title="Sensitivities")
ax3 = CairoMakie.Axis(f[2,:], xticks = 0:25:200, title="Sampling")
[plot!(ax, optsol.t, sol) for sol in eachrow(Array(optsol))[1:10]]
[plot!(ax2, optsol.t, sol) for sol in eachrow(reduce(hcat, (optsol[CorleoneCore.sensitivity_variables(layer)])))]
stairs!(ax3, 0.0:dt:last(tspan)-dt, (uopt + zero(pc)).controls[1:nc])
stairs!(ax3, 0.0:dt:last(tspan)-dt, (uopt + zero(pc)).controls[nc+1:2*nc])
stairs!(ax3, 0.0:dt:last(tspan)-dt, (uopt + zero(pc)).controls[2*nc+1:3*nc])
stairs!(ax3, 0.0:dt:last(tspan)-dt, (uopt + zero(pc)).controls[3*nc+1:4*nc])
f