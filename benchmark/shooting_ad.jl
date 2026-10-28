# Run one backend per fresh Julia process; see benchmark/README.md.
# Timings exclude package loading. First-gradient timing follows one forward call.
using Corleone, LuxCore, OrdinaryDiffEqTsit5, ComponentArrays, Random, BenchmarkTools
import DifferentiationInterface as DI
using ADTypes
const SB = Corleone.SciMLBase
const BACKEND = isempty(ARGS) ? "forwarddiff" : only(ARGS)
if BACKEND == "forwarddiff"
    using ForwardDiff
    backend = AutoForwardDiff()
elseif BACKEND == "reversediff" || BACKEND == "reversediff_default"
    using ReverseDiff, SciMLSensitivity
    backend = AutoReverseDiff()
elseif BACKEND == "zygote"
    using Zygote, SciMLSensitivity
    backend = AutoZygote()
elseif BACKEND == "mooncake"
    using Mooncake, SciMLSensitivity
    backend = AutoMooncake()
elseif BACKEND == "enzyme"
    using Enzyme, SciMLSensitivity
    backend = AutoEnzyme(; mode = Enzyme.set_runtime_activity(Enzyme.Forward), function_annotation = Enzyme.Const)
elseif BACKEND == "finitediff"
    using FiniteDiff
    backend = AutoFiniteDiff(; fdtype = Val(:central))
else
    error("Unknown backend")
end
println("Julia=", VERSION, ", threads=", Threads.nthreads(), ", backend=", BACKEND)
flush(stdout)
function rhs!(du, u, p, t)
    du[1] = u[1] - p[2] * u[1] * u[2] - 0.4 * p[1] * u[1]
    du[2] = -u[2] + p[3] * u[1] * u[2] - 0.2 * p[1] * u[2]
    du[3] = (u[1] - 1)^2 + (u[2] - 1)^2
    return nothing
end
function case_loss(joint)
    passthrough = BACKEND in ("reversediff", "mooncake", "enzyme")
    f = passthrough ? ODEFunction{true, SB.FullSpecialize}(rhs!) : ODEFunction(rhs!)
    kwargs = passthrough ? (; sensealg = SciMLSensitivity.SensitivityADPassThrough()) : (;)
    prob = ODEProblem(f, [0.5, 0.7, 0.0], (0.0, 4.0), [0.3, 1.0, 1.0]; abstol = 1.0e-10, reltol = 1.0e-10, kwargs...)
    ics = joint ? Tuple(Tunable(i; kind = Val(:variable)) => [1.25, 2.75] for i in 1:3) : ()
    layer = ShootingLayer(prob, Tsit5(), Tunable(1; kind = Val(:parameter), default = 0.3) => collect(0.5:0.5:3.5), Constant(2; kind = Val(:parameter)) => nothing, Constant(3; kind = Val(:parameter)) => nothing, ics...)
    original_ps, original_st = LuxCore.setup(MersenneTwister(42), layer)
    ps, st = joint ? add_parameter_segments!(layer, original_ps, original_st) : (original_ps, original_st)
    q0 = collect(ComponentArray(ps))
    axes = getaxes(ComponentArray(ps))
    function loss(q)
        traj, _ = layer(nothing, ComponentArray(q, axes), st)
        return last(traj.u)[3] + 0.1 * sum(abs2, shooting_constraints(traj))
    end
    return q0, loss
end
function central_gradient(f, q, h)
    return map(eachindex(q)) do i
        plus = copy(q)
        minus = copy(q)
        plus[i] += h
        minus[i] -= h
        (f(plus) - f(minus)) / (2h)
    end
end
function run_case(joint)
    name = joint ? "states_and_parameters" : "controls"
    println("START ", BACKEND, " ", name)
    flush(stdout)
    return try
        # Keep loss specialization inside the first-call timer.
        q, f = Base.invokelatest(case_loss, joint)
        original = copy(q)
        first_forward_seconds = @elapsed Base.invokelatest(f, q)
        forward_trial = @benchmark $f($q) samples = 100 evals = 1 seconds = 10
        forward_stats = median(forward_trial)
        println(
            "FORWARD,", BACKEND, ",", name, ",", first_forward_seconds, ",",
            forward_stats.time / 1.0e6, ",", forward_stats.memory, ",", forward_stats.allocs
        )
        flush(stdout)
        first_seconds = @elapsed begin
            prep = Base.invokelatest(DI.prepare_gradient, f, backend, q)
            grad = Base.invokelatest(DI.gradient, f, prep, backend, q)
        end
        reference = central_gradient(f, q, 1.0e-5)
        refined = central_gradient(f, q, 5.0e-6)
        @assert isapprox(reference, refined; rtol = 2.0e-4, atol = 2.0e-5)
        @assert isapprox(grad, refined; rtol = 2.0e-4, atol = 2.0e-5)
        max_error = maximum(abs, grad - refined)
        q2 = q .+ [0.01sin(i) for i in eachindex(q)]
        q2_original = copy(q2)
        output = similar(q2)
        DI.gradient!(f, output, prep, backend, q2)
        reference2 = central_gradient(f, q2, 5.0e-6)
        @assert isapprox(output, reference2; rtol = 2.0e-4, atol = 2.0e-5)
        max_error = max(max_error, maximum(abs, output - reference2))
        @assert q == original && q2 == q2_original
        # Time the prepared, in-place gradient path after the correctness checks.
        trial = run(@benchmarkable(DI.gradient!($f, $output, $prep, $backend, $q)); samples = 100, evals = 1, seconds = 10)
        m = median(trial)
        println("RESULT,", BACKEND, ",", name, ",PASS,", first_seconds, ",", m.time / 1.0e6, ",", m.memory, ",", m.allocs, ",", max_error)
        flush(stdout)
    catch err
        println("RESULT,", BACKEND, ",", name, ",FAIL,", replace(first(sprint(showerror, err), 300), '\n' => ' '))
        flush(stdout)
        open(joinpath(tempdir(), "corleone-shooting-ad-" * BACKEND * "-" * name * "-error.log"), "w") do io
            showerror(io, err, catch_backtrace())
        end
    end
end
run_case(false)
run_case(true)
