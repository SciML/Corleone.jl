# Efficient Julia & SciML Development Guide

## OBJECTIVE

Write idiomatic, performant, and maintainable Julia code that is compatible with automatic differentiation (Zygote/Enzyme) and SciML ecosystems.

---

## CRITICAL PRINCIPLES (Non-Negotiable)

### 1. Type Stability
```julia
# ✅ IDIOMATIC: Type-stable - same return type for all inputs
function pos(x)
    x < 0 ? zero(x) : x
end

# ❌ ANTI-PATTERN: Type-unstable - Int or Float64
function pos_bad(x)
    x < 0 ? 0 : x
end
```
**Rule**: Always return consistent types regardless of input values.

### 2. The Zygote Rule
```julia
# ✅ IDIOMATIC: Broadcasting - differentiable
function scale(x, α)
    return α .* x  # Creates new array
end

# ❌ ANTI-PATTERN: In-place mutation - breaks Zygote
function scale_bad!(x, α)
    x .= α .* x  # Mutates input
    return x
end
```
**Rule**: No in-place mutation (`.=`, `push!`, `append!`) in loss loops, ODE RHS, or any differentiable path.

### 3. Functions First, Generics by Default
```julia
# ✅ IDIOMATIC: Generic function
addone(x) = x + oneunit(x)

# ❌ ANTI-PATTERN: Too restrictive
addone_bad(x::Float64) = x + 1.0
```
**Rule**: Use abstract types (`AbstractVector`, `Number`) or omit types for maximum generality. JIT compiler will specialize.

### 4. Multiple Dispatch
```julia
# ✅ IDIOMATIC: Dispatch on types
mynorm(x::Vector) = sqrt(real(dot(x, x)))
mynorm(A::Matrix) = maximum(svdvals(A))
```
**Rule**: Use type dispatch instead of if-statements for type-specific behavior.

---

## CODE STYLE QUICK REFERENCE

### Naming
```julia
MyStruct          # Types/Modules: CamelCase
my_function       # Functions/Variables: snake_case
CONSTANT          # Constants: UPPER_SNAKE_CASE
is_valid()        # Booleans: optional ? suffix
modify!(x)        # Mutation markers: append !
```

### Formatting
- 4-space indentation (no tabs)
- 92-character line limit (soft)
- Spaces around operators: `x + y`, not `x+y`
- No trailing whitespace

### Function Definition
```julia
# SHORT: Single-line
f(x, y) = x + y

# LONG: Multi-line with return
function process_data(data::AbstractArray{T}; threshold=0.9) where {T<:Number}
    result = similar(data)
    for i in eachindex(data)
        result[i] = data[i] > threshold ? data[i] : zero(data[i])
    end
    return result
end
```

---

## PERFORMANCE PATTERNS

### 1. Broadcasting Fusion
```julia
# ✅ IDIOMATIC: Fused - no intermediate allocations
result = sin.(x) + cos.(y)
result = @. sin(x) + cos(y)  # Equivalent
```
**Why**: Operations fuse together, avoiding temporary arrays.

### 2. Views Over Copies
```julia
# ✅ IDIOMATIC: Zero-copy
subset = view(matrix, :, 1:5)
subset = @view matrix[1:10, :]

@views for i in 1:n
    process(matrix[:, i])
end

# ❌ ANTI-PATTERN: Creates copy
subset = matrix[:, 1:5]
```
**Why**: Views create references without copying memory.

### 3. Pre-allocation
```julia
# ✅ IDIOMATIC: Pre-allocate + fill
function cumulative_sum!(result::AbstractVector{T}, data::AbstractVector{T}) where T
    total = zero(T)
    for (i, val) in enumerate(data)
        total += val
        result[i] = total
    end
    return result
end

result = similar(data)
cumulative_sum!(result, data)
```

### 4. Generic Iteration
```julia
# ✅ IDIOMATIC: Works with any indexing
for i in eachindex(array)
    array[i] *= 2
end

# ❌ ANTI-PATTERN: Fails with OffsetArrays
for i in 1:length(array)
    array[i] *= 2
end
```

### 5. Avoid Globals
```julia
# ✅ IDIOMATIC: Inside function
function process_all(data)
    result = zero(eltype(data))
    for value in data
        result += value
    end
    return result
end

# ❌ ANTI-PATTERN: Slow, type-unstable
data = [1, 2, 3]
result = 0
for value in data
    result += value
end
```

---

## SCIML & LUX PATTERNS

### Lux Layer Interface
```julia
using LuxCore

struct MyLayer{T} <: LuxCore.AbstractLuxLayer
    data::T
end

# Parameters (tunable, learned)
function LuxCore.initialparameters(rng::AbstractRNG, layer::MyLayer)
    return (; data = randn(rng, size(layer.data)))
end

# States (runtime, non-tunable, reset per forward pass)
function LuxCore.initialstates(::AbstractRNG, layer::MyLayer)
    return (; cache = nothing, counter = 0)
end

# Forward pass
function LuxCore.apply(layer::MyLayer, x, ps, st)
    output = ps.data .* x .+ st.counter
    new_st = merge(st, (; counter = st.counter + 1))
    return output, new_st
end

# Initialize
rng = Random.default_rng()
layer = MyLayer(ones(3))
ps, st = LuxCore.setup(rng, layer)
```

**Key Principles**:
- **Parameters**: Tunable weights (learned during training)
- **States**: Runtime info (counters, caches, reset each evaluation)
- **Apply**: Forward pass returns `(output, new_state)`

### Type-Level Dispatch
```julia
# ✅ IDIOMATIC: Bool at type level (zero-cost)
struct ControlParameter{T, C, B, SHOOTED, N} <: LuxCore.AbstractLuxLayer
    name::N
    t::T
    controls::C
    bounds::B
    # SHOOTED is Bool at type level (true/false)
end

# Dispatch on type-level boolean
is_shooted(::ControlParameter{...,...,..., true}) = true
is_shooted(::ControlParameter{...,...,..., false}) = false
```

### Deep Mapping with Functors
```julia
using Functors

# Recursively apply function to all leaves in nested structure
get_lower_bound(layer::AbstractLuxLayer) = Functors.fmapstructure(
    Base.Fix2(to_val, -Inf),
    LuxCore.initialparameters(Random.default_rng(), layer)
)

# Works through nested NamedTuples, arrays, struct fields
```

### Selective Reconstruction
```julia
using SciMLBase

function SciMLBase.remake(layer::MyLayer; kwargs...)
    return MyLayer(
        get(kwargs, :field1, layer.field1),
        get(kwargs, :field2, layer.field2)
    )
end

# Usage
new_layer = remake(layer; field2 = 3.0)
```

### Time Binning
```julia
const MAXBINSIZE = 100

function bin_timegrid(timegrid::Vector)
    N = length(timegrid)
    partitions = collect(1:MAXBINSIZE:N)
    if isempty(partitions) || last(partitions) != N + 1
        push!(partitions, N + 1)
    end
    return ntuple(i -> timegrid[partitions[i]:(partitions[i + 1] - 1)], length(partitions) - 1)
end
```
**Why**: Prevents stack overflow in recursive problems with large sequences.

---

## TYPE SYSTEM PATTERNS

### Parametric Types
```julia
# ✅ IDIOMATIC: Flexible, type-stable
struct MyContainer{T<:AbstractFloat}
    data::Vector{T}
    scale::T
end

c1 = MyContainer([1.0, 2.0], 1.0)            # Float64
c2 = MyContainer(Float16[1, 2], Float16(1))  # Float16
```

### Abstract Type Hierarchies
```julia
abstract type AbstractSolver end

struct NewtonSolver{T<:AbstractFloat} <: AbstractSolver
    tolerance::T
    max_iter::Int
end

function solve!(problem, solver::AbstractSolver)
    # Generic implementation
end
```

### Nullable Values
```julia
function find_target(data::AbstractVector, target)
    idx = findfirst(==(target), data)
    return idx === nothing ? nothing : data[idx]
end
# Returns Union{eltype(data), Nothing}
```

---

## COMMON IDIOMS

### Keyword Arguments with @kwdef
```julia
Base.@kwdef struct SolverOptions{T<:AbstractFloat}
    tolerance::T = 1e-6
    max_iterations::Int = 1000
    verbose::Bool = false
end

options = SolverOptions(tolerance=1e-8, verbose=true)
```

### Comprehensions vs Generators
```julia
squares = [x^2 for x in 1:10]           # Array (eager)
total = sum(x^2 for x in 1:10)          # Generator (lazy, no allocation)
matrix = [i * j for i in 1:3, j in 1:4]  # Nested comprehension
```

### Multiple Return Values
```julia
function compute_stats(data)
    return mean(data), std(data), length(data)
end

m, s, n = compute_stats(data)  # Destructure

# NamedTuple for clarity
function analyze(data)
    return (mean=mean(data), std=std(data), count=length(data))
end
```

### Do-Blocks
```julia
open("data.txt", "r") do io
    data = read(io, String)
    process(parse_data(data))
end
```

---

## ANTI-PATTERNS (Avoid These)

### Type Piracy
```julia
# ❌ NEVER extend Base on types you don't own
import Base: *
*(x::Symbol, y::Symbol) = Symbol(x, y)

# ✅ Create your own method
symbol_concat(x::Symbol, y::Symbol) = Symbol(x, y)
```

### Elaborate Container Types
```julia
# ❌ Slow, confusing
a = Vector{Union{Int,AbstractString,Tuple,Array}}(undef, n)

# ✅ Use Any or specific types
a = Vector{Any}(undef, n)
a = Vector{Float64}(undef, n)
```

### Closures in Hot Paths
```julia
# ❌ Closure causes boxing
function process_closure(data)
    multiplier = 2
    return map(x -> x * multiplier, data)
end

# ✅ Explicit function or Base.Fix2
function multiply_by_two(x)
    return x * 2
end

function process_explicit(data)
    return map(multiply_by_two, data)
end

# ✅ Or use Base.Fix2
process_fix(data) = map(Base.Fix2(*, 2), data)
```

### Unnecessary Macros
```julia
# ❌ Macro as function
macro compute_square(x)
    return :($x * $x)
end

# ✅ Simple function
square(x) = x * x
```

---

## TESTING PATTERNS

### Test Structure
```julia
using Test

@testset "Math functions" begin
    @testset "Addition" begin
        @test add(1, 2) == 3
        @test add(-1, 5) == 4
    end

    @testset "Type stability" begin
        @test @inferred add(2, 3) == 3
    end

    @testset "Floating-point" for i in 1:3
        @test i * 1.0 ≈ i atol=1e-12
    end

    @testset "Error handling" begin
        @test_throws DomainError divide(1, 0)
    end
end
```

### Use @inferred for Type Stability
```julia
@test @inferred multiply(2, 3) == 6  # Fails if type-unstable
```

---

## QUICK CHECKLIST

Before deploying code, verify:

### General Julia
- [ ] Functions are type-stable (use `@inferred` in tests)
- [ ] Functions use generic types (`AbstractVector`, `Number`) or omit types
- [ ] Mutating functions end with `!` and return modified object
- [ ] Broadcasting used correctly with `.` operators or `@.`
- [ ] No globals in performance-critical code
- [ ] Views (`@view`, `view`) used instead of copies
- [ ] `eachindex` used instead of `1:length`
- [ ] Naming conventions followed (CamelCase, snake_case, UPPER_SNAKE_CASE)

### SciML / AD Code
- [ ] No in-place mutation in differentiable paths (loss loops, ODE RHS)
- [ ] Lux.jl layer interface implemented correctly (`initialparameters`, `initialstates`, `apply`)
- [ ] Parameters (tunable) vs States (runtime) properly separated
- [ ] Functional patterns used (`map`, `.` operators) instead of mutation
- [ ] ChainRulesCore `@non_differentiable` used for non-differentiable paths
- [ ] SciMLBase.remake implemented for selective updates

### Documentation
- [ ] Docstrings present for public APIs
- [ ] Docstrings follow Julia manual format (Arguments, Returns, Examples)
- [ ] Examples provided with `julia-repl` blocks
- [ ] Cross-references with `@ref`

### Performance
- [ ] Pre-allocation used in loops
- [ ] Views used for subarray slicing
- [ ] Broadcasting fusion utilized
- [ ] No unnecessary allocations in hot paths
- [ ] Type-stable struct fields (no `Any`, no small unions)

---

## REFERENCE IMPLEMENTATION

**Corleone.jl** exemplifies A+ idiomatic Julia patterns:
- Perfect Lux.jl parameter/state separation
- Masterful use of generated functions for tuple unrolling
- Type-stable operations throughout
- Full SciMLBase/AD compatibility
- SymbolicIndexingInterface integration
- Consistent broadcasting patterns

**See**: `src/controls.jl`, `src/single_shooting.jl`, `src/trajectory.jl`

---

## FURTHER RESOURCES

**Official Documentation:**
- [Julia Manual - Style Guide](https://docs.julialang.org/en/v1/manual/style-guide/)
- [Julia Manual - Performance Tips](https://docs.julialang.org/en/v1/manual/performance-tips/)
- [Julia Manual - Methods](https://docs.julialang.org/en/v1/manual/methods/)

**SciML Resources:**
- [SciML Style Guide](https://docs.sciml.ai/SciMLStyle/stable/)
- [Lux.jl Documentation](https://lux.csail.mit.edu/)
- [Zygote.jl](https://fluxml.ai/Zygote.jl/)
- [ModelingToolkit.jl](https://docs.sciml.ai/ModelingToolkit/)

**Community:**
- [BlueStyle](https://github.com/JuliaDiff/BlueStyle)
- [Julia Anti-Patterns](https://jgreener64.github.io/julia-anti-patterns/)

---

## SUMMARY

**Core Philosophy**: Leverage Julia's strengths—multiple dispatch, JIT compilation, and expressive syntax—to write code that is both fast and readable. Type stability and generality are key to performance.

**For SciML/AD**: Differentiability constraints take precedence. If a performance pattern conflicts with AD compatibility (e.g., in-place mutation), favor functional/immutable patterns for loss loops, ODE RHS functions, and any code differentiated by Zygote or Enzyme.

**Remember**: Idiomatic Julia code follows these principles consistently across all layers—from low-level performance optimizations to high-level API design.