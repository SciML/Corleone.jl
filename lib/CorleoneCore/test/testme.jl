using CorleoneCore
using Pkg
Pkg.activate(joinpath(pwd(), "../.."))

using OrdinaryDiffEqTsit5
using SciMLSensitivity
using LuxCore
using Random
using ComponentArrays

using CairoMakie
using BenchmarkTools
using Zygote
using SciMLSensitivity.ForwardDiff

function lotka_dynamics(du, u, p, t, c)
    du[1] = u[1] - prod(u) - 0.2 * c[1] * u[1]
    du[2] = -u[2] + prod(u) - 0.4 * c[2] * u[2]
end

function make_benchmark(f, Δt, tspans...)
    t0 = first(first(tspans))
    tinf = last(last(tspans))

    controls = SignalContainer(
        PiecewiseConstant(
            t0:Δt:(tinf-Δt)
        ),
        PiecewiseConstant(
            t0:Δt:(tinf-Δt)
        ),
        aggregation=Base.Fix1(reduce, vcat)
    )
    dyn = ControlledDynamics(f, controls)

    cdyn = CorleoneCore.wrap_model(Random.default_rng(), dyn)

    u0 = ones(2)
    tspan = (t0, tinf)
    p = ComponentArray(LuxCore.initialparameters(Random.default_rng(), dyn))
    problem = ODEProblem(cdyn, u0, tspan, p)
    layers = map(tspans) do tspani
        (gensym(), ShootingProblem(problem, tspan = tspani))
    end |> NamedTuple

    shootingproblems = CorleoneCore.MultipleShootingProblem(layers)
    simlayer = CorleoneCore.SolverLayer(Tsit5(), EnsembleSerial(), shootingproblems)
    ps, st = LuxCore.setup(Random.default_rng(), simlayer)
    p = ComponentVector(ps)
    loss = let prob = simlayer, dyn = problem, st = st, ax = ComponentArrays.getaxes(p)
        function (p, st=st)
            p = ComponentArray(p, ax)
            sol, _ = prob(dyn, p, st)
            Xs = map(Array, sol)
            loss = sum(1:2) do i
                sum(Xs[i][:, end] .- Xs[i+1][:, 1])
            end
            loss + sum(abs2, [sum(Xs[i] .- 1.0) for i in eachindex(Xs)])
        end
    end
    return loss, collect(p)
end


using BenchmarkTools


benchmarks = [] 

benchmarks = map([3.0, 1.5, 1.0, 0.5, 0.25, 0.1, 0.05, 0.025, 0.01]) do dt
    t0 = time()
    l, p = make_benchmark(lotka_dynamics, dt, (0., 3.), (3., 6.), (6., 9.), (9., 12.))
    l(p)
    btime = t0 - time()
    @info "Blaa"
    ftime = @btimed $l($p)
    gtime = @btimed Zygote.gradient($l, $p)
    htime = @btimed ForwardDiff.hessian($l, $p)
    hess = ForwardDiff.hessian(l, p)
    (;
        N = length(p),
        building = btime,  
        forward = ftime, 
        gradient = gtime, 
        hessian = htime,
        hessian_matrix = hess
    )
end



f = Figure()
xs = map(Base.Fix2(getproperty, :N), benchmarks)
ax = CairoMakie.Axis(f[1,1], title = "Lotka Volterra Optimal Control", xlabel = "Number of Parameters", ylabel = "Time in Seconds", 
    yscale = log10, xscale = log10)
#ys = map(x->-x.building, benchmarks)
#scatterlines!(xs, ys, label = "Building Time")
ys = map(x->x.forward.time, benchmarks)
scatterlines!(xs, ys, label = "Forward")
ys = map(x->x.gradient.time, benchmarks)
scatterlines!(xs, ys, label = "Gradient")
ys = map(x->x.hessian.time, benchmarks)
scatterlines!(xs, ys, label = "Hessian")
f[1, 2] = Legend(f, ax, "Evaluation", framevisible = false)
display(f)

spy(benchmarks[end].hessian_matrix)


xs = map(Base.Fix2(getproperty, :N), benchmarks)
ax = CairoMakie.Axis(f[1,1], title = "Lotka Volterra Optimal Control", xlabel = "Number of Parameters", ylabel = "Allocations", yscale = log10)
ys = map(x->-x.building, benchmarks)
scatterlines(xs, ys, label = "Building Time")

f = Figure()
xs = map(Base.Fix2(getproperty, :N), benchmarks)
ax = CairoMakie.Axis(f[1,1], title = "Lotka Volterra Optimal Control", xlabel = "Number of Parameters", ylabel = "Allocations", yscale = log10)
#ys = map(x->-x.building, benchmarks)
#scatterlines!(xs, ys, label = "Building Time")
ys = map(x->x.forward.alloc, benchmarks)
scatterlines!(xs, ys, label = "Forward")
ys = map(x->x.gradient.alloc, benchmarks)
scatterlines!(xs, ys, label = "Gradient")
ys = map(x->x.hessian.alloc, benchmarks)
scatterlines!(xs, ys, label = "Hessian")
f[1, 2] = Legend(f, ax, "Evaluation", framevisible = false)
display(f)

f = Figure()
xs = map(Base.Fix2(getproperty, :N), benchmarks)
ax = CairoMakie.Axis(f[1,1], title = "Lotka Volterra Optimal Control", xlabel = "Number of Parameters", ylabel = "Bytes", 
    yscale = log10, xscale = log10)
#ys = map(x->-x.building, benchmarks)
#scatterlines!(xs, ys, label = "Building Time")
ys = map(x->x.forward.bytes, benchmarks)
scatterlines!(xs, ys, label = "Forward")
ys = map(x->x.gradient.bytes, benchmarks)
scatterlines!(xs, ys, label = "Gradient")
ys = map(x->x.hessian.bytes, benchmarks)
scatterlines!(xs, ys, label = "Hessian")
f[1, 2] = Legend(f, ax, "Evaluation", framevisible = false)
display(f)


using Zygote
using BenchmarkTools



length(p)

l(p)

@btime l($p);

using SciMLSensitivity.ForwardDiff

Zygote.gradient(loss, p)

@btime Zygote.gradient($loss, $p)

@btime ForwardDiff.hessian($loss, $p)

H = ForwardDiff.hessian(loss, p)

spy(H)