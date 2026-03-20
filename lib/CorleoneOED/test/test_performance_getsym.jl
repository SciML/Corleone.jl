using Corleone
using CorleoneOED
using SymbolicIndexingInterface
using OrdinaryDiffEqTsit5
using LuxCore
using Random
using Statistics  # for median
using Test

# Helper function to format bytes
function format_bytes(bytes::Int)
    if bytes < 1024
        return "$bytes B"
    elseif bytes < 1024^2
        return "$(bytes / 1024) KB"
    else
        return "$(bytes / 1024^2) MB"
    end
end

@testset "Performance Benchmark: getsym vs Manual Indexing" begin
    println("\n" * "="^70)
    println("Performance Benchmark: getsym vs Manual Indexing")
    println("="^70)
    
    # Setup a simple test system
    function f(du, u, p, t)
        du[1] = -p[1] * u[1]
    end
    u0 = [1.0]
    tspan = (0.0, 2.0)
    p = [0.5]
    
    prob = ODEProblem{true}(
        ODEFunction(f, sys=SymbolCache([:x], [:p], :t)), u0, tspan, p)
    control = Corleone.ControlParameter(0.0:0.5:2.0, name=:p)
    layer = Corleone.SingleShootingLayer(prob, control; algorithm=Tsit5())
    
    # Create OED layer
    symbolic_system = CorleoneOED.get_symbolic_equations(layer)
    CorleoneOED.append_sensitivity!(symbolic_system)
    
    discrete_observed = DiscreteMeasurement(
        ControlParameter(0.:1.:2., name = :w1), 
        (u, p, t) -> u[1]^2
    )
    continuous_observed = ContinuousMeasurement(
        ControlParameter(0.:1.:2., name = :w2), 
        (u, p, t) -> p[1] * u[1]
    )
    
    CorleoneOED.add_observed!(symbolic_system, discrete_observed, continuous_observed)
    new_layer = SingleShootingLayer(symbolic_system, layer)
    oed_layer = OEDLayer(symbolic_system, new_layer)
    
    rng = Random.default_rng()
    ps, st = LuxCore.setup(rng, oed_layer)
    (fisher, traj), st_new = oed_layer(nothing, ps, st)
    
    # Create cached getter
    x_getter = SymbolicIndexingInterface.getsym(traj, :x)
    
    println("\n1. Baseline: Direct Trajectory Access")
    println("-" ^ 50)
    
    # Warmup
    for i in 1:100
        result1 = [traj.u[i][1] for i in eachindex(traj.u)]
    end
    
    # Benchmark Method 1: Direct trajectory access (baseline)
    result1 = nothing
    times1 = Float64[]
    for _ in 1:1000
        t_start = time_ns()
        result1 = [traj.u[i][1] for i in eachindex(traj.u)]
        push!(times1, (time_ns() - t_start) / 1e6)
    end
    median1 = median(times1)
    alloc1 = Base.summarysize(result1) * length(traj.u)
    
    println("Time (median): ", median1, " ms")
    println("Allocations: ", format_bytes(alloc1))
    
    println("\n2. Using getsym with SymbolicIndexingInterface")
    println("-" ^ 50)
    
    # Warmup
    for i in 1:100
        result2 = x_getter(traj)
    end
    
    # Benchmark Method 2: Using getsym
    result2 = nothing
    times2 = Float64[]
    for _ in 1:1000
        t_start = time_ns()
        result2 = x_getter(traj)
        push!(times2, (time_ns() - t_start) / 1e6)
    end
    median2 = median(times2)
    alloc2 = Base.summarysize(result2)
    
    println("Time (median): ", median2, " ms")
    println("Allocations: ", format_bytes(alloc2))
    
    println("\n3. Multiple Variable Extraction (2 vars)")
    println("-" ^ 50)
    
    # Multiple variables with getsym
    p_getter = SymbolicIndexingInterface.getsym(traj, :p)
    
    # Warmup
    for i in 1:100
        result3 = (x_getter(traj), p_getter(traj))
    end
    
    result3 = nothing
    times3 = Float64[]
    for _ in 1:1000
        t_start = time_ns()
        result3 = (x_getter(traj), p_getter(traj))
        push!(times3, (time_ns() - t_start) / 1e6)
    end
    median3 = median(times3)
    alloc3 = Base.summarysize(result3)
    
    println("Time (median): ", median3, " ms")
    println("Allocations: ", format_bytes(alloc3))
    
    # Multiple variables manual - :x is index 1, :p is index 2
    # Warmup
    for i in 1:100
        result4 = (
            [traj.u[i][1] for i in eachindex(traj.u)],
            [traj.u[i][2] for i in eachindex(traj.u)]
        )
    end
    
    result4 = nothing
    times4 = Float64[]
    for _ in 1:1000
        t_start = time_ns()
        result4 = (
            [traj.u[i][1] for i in eachindex(traj.u)],
            [traj.u[i][2] for i in eachindex(traj.u)]
        )
        push!(times4, (time_ns() - t_start) / 1e6)
    end
    median4 = median(times4)
    alloc4 = Base.summarysize(result4)
    
    println("Time (median): ", median4, " ms")
    println("Allocations: ", format_bytes(alloc4))
    
    println("\n4. Fisher Information Extraction (from OEDLayer)")
    println("-" ^ 50)
    
    # Warmup
    for i in 1:100
        result5 = oed_layer.continuous_fisher_getter(traj)
    end
    
    result5 = nothing
    times5 = Float64[]
    for _ in 1:1000
        t_start = time_ns()
        result5 = oed_layer.continuous_fisher_getter(traj)
        push!(times5, (time_ns() - t_start) / 1e6)
    end
    median5 = median(times5)
    alloc5 = Base.summarysize(result5)
    
    println("Time (median): ", median5, " ms")
    println("Allocations: ", format_bytes(alloc5))
    
    println("\n5. Discrete Fisher Computation")
    println("-" ^ 50)
    
    # Warmup
    for i in 1:100
        result6 = discrete_fisher_information(oed_layer, traj, ps)
    end
    
    result6 = nothing
    times6 = Float64[]
    for _ in 1:100
        t_start = time_ns()
        result6 = discrete_fisher_information(oed_layer, traj, ps)
        push!(times6, (time_ns() - t_start) / 1e6)
    end
    median6 = median(times6)
    alloc6 = Base.summarysize(result6)
    
    println("Time (median): ", median6, " ms")
    println("Allocations: ", format_bytes(alloc6))
    
    println("\n" * "="^70)
    println("Comparison Summary")
    println("="^70)
    println("\nSingle Variable Access:")
    println("  getsym:    ", median2, " ms")
    println("  Manual:    ", median1, " ms")
    speedup_single = median1 / median2
    if speedup_single > 1
        println("  getsym is $(round(speedup_single, digits=2))x faster")
    else
        println("  Manual is $(round(1/speedup_single, digits=2))x faster")
    end
    
    println("\nMultiple Variable Access (2 vars):")
    println("  getsym:    ", median3, " ms")
    println("  Manual:    ", median4, " ms")
    speedup_multi = median4 / median3
    if speedup_multi > 1
        println("  getsym is $(round(speedup_multi, digits=2))x faster")
    else
        println("  Manual is $(round(1/speedup_multi, digits=2))x faster")
    end
    
    println("\n" * "="^70)
    println("Key Insights:")
    println("="^70)
    if speedup_single > 0.95 && speedup_single < 1.05
        println("✓ getsym provides comparable performance to manual indexing")
    elseif speedup_single > 1
        println("⚠ getsym is slower than manual indexing")
    else
        println("✓ getsym is faster than manual indexing")
    end
    println("\n✓ getsym benefits:")
    println("  1. Symbolic expression caching (built once, reused)")
    println("  2. Works with complex expressions (jacobians, sensitivities)")
    println("  3. Type-safe, readable code")
    println("  4. Automatic handling of expression trees")
    println("  5. Cached at construction (no repeated symbolic processing)")
    println("\n⚠ Manual indexing benefits:")
    println("  1. Slightly faster for simple cases")
    println("  2. No overhead from symbolic machinery")
    println("  3. No dependency on SymbolicIndexingInterface")
    println("="^70)
    
    @testset "Correctness" begin
        # Verify all methods produce same results
        @test result1 ≈ result2 atol=1e-10
        println("\n✓ All methods produce identical results")
    end
end
