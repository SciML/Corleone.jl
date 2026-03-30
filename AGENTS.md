# Prompt: Teaching Agents to Write Idiomatic Julia Code

**Objective**: Enable AI agents to write idiomatic, performant, and maintainable Julia code by understanding the language's core design principles and community conventions.

---

## CORE PRINCIPLES OF IDIOMATIC JULIA

### 1. Functions First, Generics by Default
- **Write reusable functions, not procedural scripts** — Julia's JIT compiler optimizes functions
- **Use abstract types for arguments** (`AbstractVector`, `Number`) or omit types entirely for maximum generality
- **Let the compiler specialize** — type annotations are for dispatch, not performance optimization

```julia
# IDIOMATIC
addone(x::Number) = x + oneunit(x)  # Generic, works with any numeric type
addone(x) = x + oneunit(x)           # Even more generic

# ANTI-PATTERN
addone(x::Float64) = x + 1.0         # Too restrictive, loses generality
```

### 2. Multiple Dispatch as a Design Tool
- **Dispatch on types** to provide type-specific behavior without if-statements
- **Extend Base methods** for your custom types to integrate with Julia's ecosystem
- **Parametric methods** capture type information for compile-time optimizations

```julia
# IDIOMATIC: Dispatch on types
mynorm(x::Vector) = sqrt(real(dot(x, x)))
mynorm(A::Matrix) = maximum(svdvals(A))

# IDIOMATIC: Parametric method
same_type(x::T, y::T) where {T} = true
same_type(x, y) = false
```

### 3. Type Stability Matters
- **Return consistent types** regardless of input values — the compiler generates faster code
- **Avoid abstract types in struct fields** — use parametric types instead
- **Use `@inferred` in tests** to catch type instability

```julia
# IDIOMATIC: Type-stable
function pos(x)
    x < 0 ? zero(x) : x  # Always returns same type as input
end

# ANTI-PATTERN: Type-unstable
function pos_bad(x)
    x < 0 ? 0 : x  # Could return Int or Float64
end

# IDIOMATIC: Type-stable struct
struct MyContainer{T<:Number}
    data::Vector{T}
    cache::T
end

# ANTI-PATTERN: Type-unstable struct
struct MyContainerBad
    data::AbstractVector{T} where T
    cache::Number
end
```

---

## CODE STYLE CONVENTIONS

### Naming Rules
- **Types/Modules**: `CamelCase` (`DataFrame`, `LinearAlgebra`)
- **Functions/Variables**: `snake_case` (`compute_mean`, `has_data`)
- **Constants**: `UPPER_SNAKE_CASE` (`MAX_ITERATIONS`)
- **Boolean functions**: End with `?` (optional, but common: `is_valid?`)
- **Mutation marker**: Append `!` to function names that modify arguments (`sort!`, `push!`)

### Formatting
- **4-space indentation** (no tabs)
- **92-character line limit** (soft guideline)
- **Spaces around operators**: `x + y`, not `x+y`
- **No trailing whitespace**
- **Blank lines between top-level functions**

### Function Definition Style
```julia
# SHORT: Single-line form
f(x, y) = x + y

# LONG: Multi-line with return
function process_data(
    data::AbstractArray{T};
    threshold::Real=0.9,
    verbose::Bool=false
) where {T<:Number}
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
**Use dot operators and `@.` for efficient element-wise operations**

```julia
# IDIOMATIC: Fused broadcasting (no intermediate allocations)
result = @. sin(x) + cos(y)

# EQUIVALENT: Dotted form
result = sin.(x) + cos.(y)

# ANTI-PATTERN: Creates intermediate arrays
result = sin(x) + cos(y)  # Fails for arrays
```

**Why**: Broadcasting fuses operations automatically, avoiding temporary array allocations. Works seamlessly with GPU arrays, OffsetArrays, and custom array types.

### 2. Pre-allocation and In-place Operations
**Pre-allocate output arrays and use mutating functions**

```julia
# IDIOMATIC: Pre-allocate + in-place update
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

# ANTI-PATTERN: Repeated allocations
function cumulative_sum_slow(data)
    return [sum(data[1:i]) for i in 1:length(data)]  # O(n²) allocations
end
```

### 3. Avoid Globals
**Keep performance-critical code inside functions**

```julia
# IDIOMATIC: Fast
function process_all(data)
    result = zero(eltype(data))
    for value in data
        result += value
    end
    return result
end

# ANTI-PATTERN: Slow, type-unstable
data = [1, 2, 3]
result = 0
for value in data  # Runs at script-level, not compiled
    result += value
end
```

### 4. Views Over Copies
**Use `view`, `@view`, or `@views` to avoid allocations**

```julia
# IDIOMATIC: Zero-copy view
subset = view(matrix, :, 1:5)
subset2 = @view matrix[1:10, :]

@views for i in 1:n
    process(matrix[:, i])  # All slices become views
end

# ANTI-PATTERN: Creates copies
subset = matrix[:, 1:5]  # Allocates new array
```

### 5. Generic Iteration with `eachindex`
**Use `eachindex` instead of `1:length` for compatibility**

```julia
# IDIOMATIC: Works with any indexing scheme
for i in eachindex(array)
    array[i] *= 2
end

# ANTI-PATTERN: Fails with OffsetArrays
for i in 1:length(array)
    array[i] *= 2  # Assumes 1-based indexing
end
```

---

## TYPE SYSTEM BEST PRACTICES

### 1. Parametric Types for Flexibility
**Use `where` clauses to constrain types while maintaining generality**

```julia
# IDIOMATIC: Parametric with constraints
struct MyContainer{T<:AbstractFloat}
    data::Vector{T}
    scale::T
end

# Works with any floating-point type
c1 = MyContainer([1.0, 2.0], 1.0)          # Float64
c2 = MyContainer(Float16[1, 2], Float16(1))  # Float16
```

### 2. Abstract Type Hierarchies
**Create interface types for polymorphism**

```julia
# IDIOMATIC: Abstract interface
abstract type AbstractSolver end

struct NewtonSolver{T<:AbstractFloat} <: AbstractSolver
    tolerance::T
    max_iter::Int
end

struct GradientSolver{T<:AbstractFloat} <: AbstractSolver
    learning_rate::T
    epochs::Int
end

# Works with any solver type
function solve!(problem, solver::AbstractSolver)
    # Generic implementation
end
```

### 3. Type Unions for Optional Values
**Use `Union{T, Nothing}` for nullable values**

```julia
# IDIOMATIC: Nullable return
function find_target(data::AbstractVector, target)
    idx = findfirst(==(target), data)
    return idx === nothing ? nothing : data[idx]
end

# Returns `Union{eltype(data), Nothing}`
```

### 4. Trait-Based Dispatch
**Use traits for compile-time decisions without runtime overhead**

```julia
# Define traits
abstract type IterationStyle end
struct IndexedIteration <: IterationStyle end
struct SequentialIteration <: IterationStyle end

# Trait function
iteration_style(::Type{<:AbstractArray}) = IndexedIteration()
iteration_style(::Type{<:AbstractSet}) = SequentialIteration()

# Dispatch on traits
function process(data, ::IndexedIteration)
    for i in eachindex(data)
        process_element(data[i])
    end
end

function process(data, ::SequentialIteration)
    for elem in data
        process_element(elem)
    end
end

# Public API
process(data) = process(data, iteration_style(typeof(data)))
```

---

## COMMON IDIOMS WITH EXAMPLES

### 1. Keyword Arguments with `@kwdef`
**Convenient struct initialization with defaults**

```julia
Base.@kwdef struct SolverOptions{T<:AbstractFloat}
    tolerance::T = 1e-6
    max_iterations::Int = 1000
    verbose::Bool = false
end

# Create with specific options
options = SolverOptions(tolerance=1e-8, verbose=true)
```

### 2. Comprehensions vs Generators
**Comprehensions for eager evaluation, generators for lazy iteration**

```julia
# IDIOMATIC: Comprehension (eager)
squares = [x^2 for x in 1:10]  # Creates array

# IDIOMATIC: Generator (lazy)
total = sum(x^2 for x in 1:10)  # No intermediate array

# Nested comprehensions
matrix = [i * j for i in 1:3, j in 1:4]
```

### 3. Multiple Return Values
**Return tuples for multiple values, destructure on call**

```julia
# IDIOMATIC
function compute_stats(data)
    return mean(data), std(data), length(data)
end

m, s, n = compute_stats(data)

# Named return with NamedTuple
function analyze(data)
    return (mean=mean(data), std=std(data), count=length(data))
end
```

### 4. Do-Blocks for Multi-line Anonymous Functions
**More readable than nested anonymous functions**

```julia
# IDIOMATIC
open("data.txt", "r") do io
    data = read(io, String)
    process(parse_data(data))
end

# EQUIVALENT but less readable
open("data.txt", "r") do io
    data = read(io, String)
    parse_data(data) |> process
end
```

### 5. Function Barriers
**Separate type-unstable setup from type-stable kernel**

```julia
# IDIOMATIC: Function barrier pattern
function process_mixed(data::Vector{Any})
    T = eltype(first(data))  # Type-unstable setup
    result = similar(data, T)

    # Type-stable kernel
    return process_kernel!(result, data)
end

@inline function process_kernel!(result::Vector{T}, data) where T
    for (i, val) in enumerate(data)
        result[i] = convert(T, val)^2
    end
    return result
end
```

---

## ANTI-PATTERNS TO AVOID

### 1. Type Piracy
**Never extend Base methods on types you don't own**

```julia
# ANTI-PATTERN: Type piracy
import Base: *
*(x::Symbol, y::Symbol) = Symbol(x, y)  # Don't do this!

# IDIOMATIC: Create your own method
symbol_concat(x::Symbol, y::Symbol) = Symbol(x, y)
```

### 2. Elaborate Container Types
**Avoid complex union types in containers**

```julia
# ANTI-PATTERN: Slow, confusing
a = Vector{Union{Int,AbstractString,Tuple,Array}}(undef, n)

# IDIOMATIC: Use Any or specific types
a = Vector{Any}(undef, n)  # Faster for heterogeneous data
a = Vector{Float64}(undef, n)  # When homogeneous
```

### 3. Closures in Hot Paths
**Closures can cause accidental type instabilities**

```julia
# ANTI-PATTERN: Closure causes boxing
function process_closure(data)
    multiplier = 2
    return map(x -> x * multiplier, data)  # Closure
end

# IDIOMATIC: Explicit function
function multiply_by_two(x)
    return x * 2
end

function process_explicit(data)
    return map(multiply_by_two, data)
end

# IDIOMATIC: Or use Base.Fix2
function process_fix(data)
    return map(Base.Fix2(*, 2), data)
end
```

### 4. Unnecessary Macros
**Use functions unless syntactic transformation is needed**

```julia
# ANTI-PATTERN: Macro as function
macro compute_square(x)
    return :($x * $x)
end

# IDIOMATIC: Simple function
square(x) = x * x
```

---

## DOCUMENTATION CONVENTIONS

### Docstring Format
**Place docstrings immediately before definitions**

```julia
"""
    process(data::AbstractArray{T}; threshold::Real=0.9) where {T<:Number}

Process input data by applying thresholding.

# Arguments
- `data::AbstractArray{T}`: Input data array
- `threshold::Real`: Threshold value (default: 0.9)

# Returns
- `Vector{T}`: Processed data

# Throws
- `ArgumentError`: if threshold is not in [0, 1]

# Examples
```julia-repl
julia> process([0.5, 0.9, 1.2], threshold=0.8)
2-element Vector{Float64}:
    0.0
    0.9
    1.2
```

See also: [`process!`](@ref), [`normalize`](@ref)
"""
function process(data::AbstractArray{T}; threshold::Real=0.9) where {T<:Number}
    # implementation
end
```

### Documentation Guidelines
- Use **4-space indent** for function signature
- Use **imperative form** ("Compute" not "Computes")
- Include **code examples** with `julia-repl` blocks
- Use **backticks** for code identifiers: `` `process(x)` ``
- Add **cross-references** with `@ref`
- Include **`# Implementation` section** for API guidance

---

## TESTING PATTERNS

### Test Structure
**Use `@testset` for hierarchical organization**

```julia
using Test

@testset "Math functions" begin
    @testset "Addition" begin
        @test add(1, 2) == 3
        @test add(-1, 5) == 4
    end

    @testset "Multiplication" begin
        @test multiply(2, 3) == 6
        @test @inferred multiply(2, 3) == 6  # Type stability check
    end

    @testset "Floating-point" for i in 1:3
        @test i * 1.0 ≈ i atol=1e-12
    end

    @testset "Error handling" begin
        @test_throws DomainError divide(1, 0)
    end
end
```

### Testing Best Practices
- **Test type coverage** with different numeric types (`Int`, `Float64`, `Complex`)
- **Use `@inferred`** to catch type instability
- **Use `≈` (`\approx`)** for floating-point comparisons with `rtol`/`atol`
- **Make tests self-contained** and **runnable**
- **Use `@test_broken`** for known failing tests

---

## PRACTICAL EXAMPLE: IDIOMATIC JULIA FUNCTION

Here's a complete, idiomatic Julia function demonstrating multiple principles:

```julia
"""
    compute_statistics!(
        result::NamedTuple,
        data::AbstractArray{T};
        weights::Union{AbstractVector{T}, Nothing}=nothing,
        normalize::Bool=true
    ) where {T<:AbstractFloat}

Compute weighted statistics and store results in-place.

# Arguments
- `result::NamedTuple`: Output storage with fields `mean`, `std`, `count`
- `data::AbstractArray{T}`: Input data array
- `weights::Union{AbstractVector{T}, Nothing}`: Optional weights (default: nothing)
- `normalize::Bool`: Whether to normalize weighted statistics (default: true)

# Returns
- `NamedTuple`: The result object for chaining

# Examples
```julia-repl
julia> result = (mean=0.0, std=0.0, count=0);
julia> compute_statistics!(result, [1.0, 2.0, 3.0]);
julia> result.mean
2.0
```
"""
function compute_statistics!(
    result::NamedTuple,
    data::AbstractArray{T};
    weights::Union{AbstractVector{T}, Nothing}=nothing,
    normalize::Bool=true
) where {T<:AbstractFloat}
    # Validate inputs early with context
    n = length(data)
    n == 0 && return result

    if weights !== nothing
        length(weights) == n || throw(ArgumentError(
            "weights length $(length(weights)) must match data length $n"
        ))
        all(w -> w >= 0, weights) || throw(ArgumentError("weights must be non-negative"))
    end

    # Type-stable kernel
    w_total = if weights === nothing
        T(n)
    else
        T(sum(weights))
    end

    # Compute weighted mean
    @views function compute_weighted_mean()
        if weights === nothing
            return sum(data) / w_total
        else
            return sum(data .* weights) / w_total
        end
    end

    result.mean = compute_weighted_mean()

    # Compute weighted standard deviation
    if n > 1
        weighted_sum_sq = if weights === nothing
            sum(@. (data - result.mean)^2)
        else
            sum(@. weights * (data - result.mean)^2)
        end

        dof = normalize ? w_total - (weights === nothing ? T(1) : T(0)) : w_total
        result.std = sqrt(weighted_sum_sq / max(dof, one(T)))
    else
        result.std = zero(T)
    end

    result.count = n

    return result  # Return for method chaining
end
```

**Why this is idiomatic:**
1. ✅ **Generic type parameter** `T<:AbstractFloat` works with any floating-point type
2. ✅ **Abstract array argument** accepts any array-like type
3. ✅ **In-place mutation** with `!` suffix, returns `result` for chaining
4. ✅ **Early validation** with context-rich error messages
5. ✅ **Type-stable computation** within function
6. ✅ **Broadcasting fusion** with `@.` for no-temporary allocations
7. ✅ **Views** for subarray referencing
8. ✅ **Comprehensive docstring** with examples
9. ✅ **Keyword arguments** for optional parameters with defaults

---

## SCIML & LUX LAYER PATTERNS

### Lux Layer Abstract Interface
When working with the **Lux** ecosystem, implement the layer interface consistently:

```julia
# ✅ IDIOMATIC: Lux layer with explicit parameter/state separation
using LuxCore

struct MyLayer{T} <: LuxCore.AbstractLuxLayer
    data::T
end

# Initialize parameters (tunable values)
function LuxCore.initialparameters(rng::Random.AbstractRNG, layer::MyLayer)
    return (; data = randn(rng, size(layer.data)))
end

# Initialize runtime state (non-tunable, evolves during evaluation)
function LuxCore.initialstates(::Random.AbstractRNG, layer::MyLayer)
    return (; cache = nothing, counter = 0)
end

# Apply the layer (forward pass)
function LuxCore.apply(layer::MyLayer, x, ps, st)
    output = ps.data .* x .+ st.counter
    new_st = merge(st, (; counter = st.counter + 1))
    return output, new_st
end

# Setup combines parameters and states
rng = Random.default_rng()
layer = MyLayer(ones(3))
ps, st = LuxCore.setup(rng, layer)
```

**Key principles**:
- **Parameters**: Tunable weights, learned during training
- **States**: Runtime information (counters, caches), reset each forward pass
- **Apply**: Forward pass returning `(output, new_state)`
- **Setup**: Initialize both parameters and states

### Lux Layer Types (Hierarchical)

```julia
# ✅ IDIOMATIC: Use appropriate Lux layer type

# Simple layer with fields
struct MyLayer <: LuxCore.AbstractLuxLayer
    field1::Type1
    field2::Type2
end

# Wrapper layer (delegates to inner layer)
struct MyWrapper{L <: AbstractLuxLayer} <: LuxCore.AbstractLuxWrapperLayer{:inner_field}
    inner_field::L
    metadata::NamedTuple
end

# Container layer (holds multiple layers)
struct MyContainer{LAYERS} <: LuxCore.AbstractLuxContainerLayer{LAYERS}
    layer1::AbstractLuxLayer
    layer2::AbstractLuxLayer
end
```

### Parametric Types for Type Stability

```julia
# ✅ IDIOMATIC: Capture all types as parameters
struct ControlParameter{T, C, B, SHOOTED, N} <: LuxCore.AbstractLuxLayer
    name::N
    t::T                    # Time grid type
    controls::C             # Controls function type
    bounds::B               # Bounds function type
    # SHOOTED is Bool at type level (true/false)
end

# Dispatch on type-level booleans
is_shooted(::ControlParameter{...,...,..., true}) = true
is_shooted(::ControlParameter{...,...,..., false}) = false
```

### Generated Functions for Performance

```julia
# ✅ IDIOMATIC: Use @generated for compile-time unrolling
@generated function process_tuple(x::Tuple{T, Vararg{T, N}}) where {T, N}
    exprs = Expr[:(
        println("Element ", $(i), ": ", x[$(i)])
    ) for i in 1:N]
    push!(exprs, :(return nothing))
    return Expr(:block, exprs...)
end

# Runtime equivalent would be:
process_tuple_loop(x::Tuple) = foreach(eachindex(x)) do i
    println("Element ", i, ": ", x[i])
end

# Generated version eliminates loop overhead at compile time
```

### Runtime-Generated Functions for Symbolic Code

```julia
# ✅ IDIOMATIC: Generate functions from symbolic expressions
using RuntimeGeneratedFunctions

function build_observer(objective_expr)
    # Construct expression AST at runtime
    expr = Expr(:function, :(trajectory), Expr(:block, :(
        return $(objective_expr)
    )))
    # Compile into a function
    return @RuntimeGeneratedFunction(expr)
end

# Usage
obj_func = build_observer(:(sum(x.^2) + sqrt(sum(u.^2))))
result = obj_func(trajectory)
```

### SciMLBase Methods: Remake Pattern

```julia
# ✅ IDIOMATIC: Implement remake for parameter modification
function SciMLBase.remake(layer::MyLayer; kwargs...)
    new_field1 = get(kwargs, :field1, layer.field1)
    new_field2 = get(kwargs, :field2, layer.field2)
    return MyLayer(new_field1, new_field2)
end

# Usage
new_layer = remake(layer; field1 = new_value)
```

**Pattern**: Allow selective reconstruction without copying all fields.

### AD Compatibility with ChainRulesCore

```julia
# ✅ IDIOMATIC: Mark non-differentiable paths
using ChainRulesCore

# Mark entire function as non-differentiable
ChainRulesCore.@non_differentiable function _non_ad_path(x, y, z)
    # Some side-effect only code
    return nothing
end

# For partial AD, implement custom rrule
function ChainRulesCore.rrule(::typeof(my_observable), layer, x, ps, st)
    # Forward pass
    y, st_new = my_observable(layer, x, ps, st)

    # Backward pass (gradient)
    function my_observable_pullback(ȳ)
        # ȳ is upstream gradient
        # Compute gradients w.r.t. ps only
        ∂ps = compute_gradient_wrt_params(ȳ, x, ps)
        return NoTangent(), ∂ps  # No gradient for layer, x
    end
    return y, my_observable_pullback
end
```

### SymbolicIndexingInterface Integration

```julia
# ✅ IDIOMATIC: Implement symbolic indexing for custom trajectories
using SymbolicIndexingInterface

struct MyTrajectory{S, U, T} <: SomeTimeseriesInterface
    sys::S                # Symbolic container
    u::U                  # State trajectory
    t::T                  # Time vector
end

# Declare as timeseries
SymbolicIndexingInterface.is_timeseries(::Type{<:MyTrajectory}) = Timeseries()

# Implement required interface
SymbolicIndexingInterface.symbolic_container(traj::MyTrajectory) = traj.sys
SymbolicIndexingInterface.state_values(traj::MyTrajectory) = traj.u
SymbolicIndexingInterface.current_time(traj::MyTrajectory) = traj.t
SymbolicIndexingInterface.parameter_values(traj::MyTrajectory) = traj.p

# Optional: observed variables (time-dependent functions)
SymbolicIndexingInterface.observed(traj::MyTrajectory, sym) = (u, p, t) -> ...
```

### Functional Composition Patterns

```julia
# ✅ IDIOMATIC: Use Base.Fix1 and Base.Fix2 for partial application
using Base

# Partial application: fix first argument
add_to_all = Base.Fix1(map, x -> x + 1)
result = add_to_all([1, 2, 3])  # [2, 3, 4]

# Partial application: fix second argument
clamp_to_range = Base.Fix2(clamp, (-1.0, 1.0))
result = clamp_to_range([0.5, -2.0, 1.5])  # [0.5, -1.0, 1.0]

# Higher-order composition
transform_data = compose(Base.Fix1(map, sqrt), Base.Fix2(filter, x -> x > 0))
result = transform_data(0:10)  # sqrt of positive values
```

### Broadcasting for AD-Friendly Operations

```julia
# ✅ IDIOMATIC: Broadcast instead of mutate
function scale_vector_ad_compatible(x::AbstractVector, α::Real)
    return α .* x  # Broadcasting - differentiable
end

# ❌ ANTI-PATTERN: In-place mutation - breaks AD
function scale_vector_bad!(x::AbstractVector, α::Real)
    x .= α .* x  # Mutation - not differentiable with Zygote
    return x
end
```

### Named Tuples as Structured Containers

```julia
# ✅ IDIOMATIC: Use NamedTuple for heterogeneous data
result = (;
    u = [1.0, 2.0, 3.0],
    t = [0.0, 1.0, 2.0],
    control = (; u = [0.5], v = [0.8])
)

# Access with getproperty (dot syntax)
val_u = result.u
val_control_u = result.control.u

# Merge for functional updates
new_result = merge(result, (; u = [4.0, 5.0, 6.0]))

# Structural typing with field names
function process_data(data::NamedTuple{:u, :t})
    # Only accepts structs with exactly these fields
    return data.u .+ data.t
end
```

### Time Binning for Recursive Problems

```julia
# ✅ IDIOMATIC: Partition large sequences to avoid stack overflow
const MAXBINSIZE = 100

function bin_timegrid(timegrid::Vector)
    N = length(timegrid)
    partitions = collect(1:MAXBINSIZE:N)
    if isempty(partitions) || last(partitions) != N + 1
        push!(partitions, N + 1)
    end
    bins = ntuple(i -> timegrid[partitions[i]:(partitions[i + 1] - 1)], length(partitions) - 1)
    return bins  # Returns tuple of sub-arrays
end

# Process bins recursively with flat tuples
function process_sequence(layer, data::Tuple)
    current, rest = Base.first(data), Base.tail(data)
    out, st = layer(current)
    if length(rest) == 0
        return (out,), st
    else
        out_rest, st_rest = process_sequence(layer, rest)
        return ((out..., out_rest...),), st_rest
    end
end
```

### Type-Level Flags with Multiple Dispatch

```julia
# ✅ IDIOMATIC: Encode configuration at type level
abstract type OptimizerAlgorithm end
struct SGD <: OptimizerAlgorithm end
struct Adam <: OptimizerAlgorithm end

struct Optimizer{A <: OptimizerAlgorithm, T <: AbstractFloat}
    algorithm::A
    learning_rate::T
end

# Dispatch on algorithm type
function step!(opt::Optimizer{SGD}, params, gradients)
    return params .- opt.learning_rate .* gradients
end

function step!(opt::Optimizer{Adam}, params, gradients)
    # Adam-specific logic
    return params .- update
end
```

### Unicode for Mathematical Notation

```julia
# ✅ IDIOMATIC: Use Unicode for clear scientific notation
function gradient_descentα(∇f::Function, x₀::AbstractVector{<:Real};
                         α::Real=0.01, max_iter::Int=1000)
    x = copy(x₀)
    for i in 1:max_iter
        ∇ = ∇f(x)
        norm(∇) < 1e-8 && break
        x .= x .- α .* ∇
    end
    return x
end

# Physical system parameters
struct HarmonicOscillator{T<:Real}
    ω²::T      # Angular frequency squared
    γ::T       # Damping coefficient
    θ₀::T      # Initial angle
    ω₀::T      # Initial angular velocity
end
```

### Documentation with DocStringExtensions

```julia
# ✅ IDIOMATIC: Use DocStringExtensions templates for consistent docs
using DocStringExtensions

"""
$(TYPEDEF)

A struct for parameterized control discretization.

# Fields
$(FIELDS)

# Examples
```julia
control = ControlParameter(0.0:0.1:10.0)
```
"""
struct ControlParameter{T, C, B, SHOOTED, S} <: LuxCore.AbstractLuxLayer
    name::S
    t::T
    controls::C
    bounds::B
end

"""
$(SIGNATURES)

Return elementwise lower bounds for layer.

# Returns
- `Vector{T}`: Lower bounds matching parameter structure
"""
function get_lower_bound(layer::AbstractLuxLayer)
    # implementation
end
```

**Why this is idiomatic**:
- `$(TYPEDEF)` automatically renders struct definition
- `$(FIELDS)` automatically lists all fields with types
- `$(SIGNATURES)` renders function signature
- Consistent with SciML ecosystem documentation

### Deep Mapping with Functors.fmapstructure

```julia
# ✅ IDIOMATIC: Recursively apply functions to parameter structures
using Functors

# Apply -Inf to all leaf elements in parameter structure
function get_lower_bound(layer::AbstractLuxLayer)
    return Functors.fmapstructure(
        Base.Fix2(to_val, -Inf),
        LuxCore.initialparameters(Random.default_rng(), layer)
    )
end

# Apply Inf to all leaf elements
function get_upper_bound(layer::AbstractLuxLayer)
    return Functors.fmapstructure(
        Base.Fix2(to_val, Inf),
        LuxCore.initialparameters(Random.default_rng(), layer)
    )
end

# Convert scalar conversion function to deep structure
fmap(Base.Fix2(_to_val, Float32), parameters)  # Convert all to Float32
```

**Why this is idiomatic**:
- Works recursively through nested `NamedTuple`, fields, arrays
- Preserves structure exactly
- No manual recursion needed
- Standard pattern in Lux.jl ecosystem

### Type-Stable Array Reconstruction with Boolean Masks

```julia
# ✅ IDIOMATIC: Use boolean matrices for selective array updates
function prepare_u0_state(u0::AbstractVector, tunable_ic::Vector{Int})
    # Boolean mask for fixed positions
    keeps = [i ∉ tunable_ic for i in eachindex(u0)]

    # Boolean replacement matrix (one-hot encoding)
    replaces = zeros(Bool, length(u0), length(tunable_ic))
    for (i, idx) in enumerate(tunable_ic)
        replaces[idx, i] = true
    end

    return (; u0 = copy(u0), keeps, replaces)
end

function apply_parameters(u0_fixed, keeps, replaces, tunable_params)
    # keeps .* u0_fixed keeps fixed positions unchanged
    # replaces * tunable_params places tunable values in correct positions
    u0_new = keeps .* u0_fixed .+ replaces * tunable_params
    return u0_new
end

# Usage
state = prepare_u0_state([1.0, 2.0, 3.0, 4.0], [1, 3])
u0_new = apply_parameters(state.u0, state.keeps, state.replaces, [5.0, 6.0])
# Result: [5.0, 2.0, 6.0, 4.0]
```

**Why this is idiomatic**:
- Type-stable: always returns same type as input `u0`
- No conditional branching (no `if-else` inside loop)
- Matrix-vector multiply is highly optimized
- AD-friendly (no mutation of parameters)
- Works for any array type and any subset of indices

### Selective Reconstruction with Remake

```julia
# ✅ IDIOMATIC: Implement remake for selective field updates
using SciMLBase

function SciMLBase.remake(layer::MyLayer; kwargs...)
    # Extract fields with fallback to original
    new_field1 = get(kwargs, :field1, layer.field1)
    new_field2 = get(kwargs, :field2, layer.field2)
    # Fields not in kwargs keep original values
    return MyLayer(new_field1, new_field2)
end

# Pattern with nested layers
function SciMLBase.remake(layer::MultiLayer; kwargs...)
    # Specific sub-layer modifications
    new_sublayer1 = get(kwargs, :sublayer1, kwargs) do _
        remake(layer.sublayer1; kwargs...)
    end

    # Keep other sub-layers as-is
    sublayer2 = layer.sublayer2

    # Top-level fields
    name = get(kwargs, :name, layer.name)

    return MultiLayer(name, new_sublayer1, sublayer2)
end

# Usage
original = MyLayer(field1=1.0, field2=2.0)
modified = remake(original; field2=3.0)
# Result: MyLayer(field1=1.0, field2=3.0)
```

**Why this is idiomatic**:
- Functional: returns new object, doesn't mutate
- Default to original values with `get`
- Forward kwargs to nested layers
- Standard SciMLBase API

### Type-Stable Time Grid Logic

```julia
# ✅ IDIOMATIC: Handle edge cases with type-stable expressions
function get_timegrid_or_default(t::AbstractVector)
    return isempty(t) ? (0.0, 0.0) : extrema(t)
end

# For empty time grids, return sensible defaults
# extrema throws on empty vectors, so we handle it early
t_empty = Float64[]
t_full = [0.0, 1.0, 2.0]

get_timegrid_or_default(t_empty)   # (0.0, 0.0)
get_timegrid_or_default(t_full)    # (0.0, 2.0)
```

**Why this is idiomatic**:
- Type-stable: always returns `Tuple{Float64, Float64}`
- Early return avoids type assertion failures
- Clear error handling without exceptions

---

## SUMMARY CHECKLIST

### General Julia Code
When writing Julia code, always ask:

- [ ] **Is my code type-stable?** (Same return type regardless of input values)
- [ ] **Am I using abstract types appropriately?** (`AbstractVector` over `Vector`, `Number` over `Float64`)
- [ ] **Do my mutating functions return the modified object?** (For chaining)
- [ ] **Am I using broadcasting efficiently?** (`@.` for fusion)
- [ ] **Am I avoiding allocations in hot loops?** (Pre-allocate, use views, generators)
- [ ] **Do my error messages provide context and guidance?**
- [ ] **Is my code generic and reusable?** (Avoid overly-specific constraints)
- [ ] **Am I following naming conventions?** (`!` for mutation, snake_case for functions)
- [ ] **Do I have comprehensive docstrings with examples?**
- [ ] **Are my functions small and focused?** (Dispatch handles specialization)

### SciML & Automatic Differentiation (AD) Code
**Additional checklist for differentiable scientific computing:**

- [ ] **Is my code Zygote/Enzyme compatible?** (No hidden mutations in AD paths)
- [ ] **Am I using broadcasting instead of in-place operations?** (`x .+ y` not `x .= y`)
- [ ] **Do I return new structures instead of mutating?** (Functional patterns for AD)
- [ ] **Are all struct fields concretely typed?** (No `Any`, no small unions)
- [ ] **Am I using Lux.jl over Flux.jl for neural networks?** (Explicit state)
- [ ] **Do I handle `Num` types correctly in ModelingToolkit?** (Use `Symbolics.unwrap`)
- [ ] **Am I dispatching on appropriate abstract types?** (`AbstractSystem`, `AbstractVector{<:Real}`)
- [ ] **Do I use Unicode for mathematical notation?** (α, β, ∇, etc., when clear)
- [ ] **Have I provided custom `rrule`s if mutation is unavoidable?** (ChainRulesCore)
- [ ] **Is my ODE system defined with `@mtkmodel`?** (v9+ declarative syntax)

---

## FURTHER LEARNING

**Official Documentation:**
- [Julia Manual - Style Guide](https://docs.julialang.org/en/v1/manual/style-guide/)
- [Julia Manual - Performance Tips](https://docs.julialang.org/en/v1/manual/performance-tips/)
- [Julia Manual - Methods](https://docs.julialang.org/en/v1/manual/methods/)
- [Julia Manual - Types](https://docs.julialang.org/en/v1/manual/types/)
- [Julia Manual - Documentation](https://docs.julialang.org/en/v1/manual/documentation/)

**Community Resources:**
- [SciML Style Guide](https://docs.sciml.ai/SciMLStyle/stable/) - Scientific Computing Patterns
- [BlueStyle](https://github.com/JuliaDiff/BlueStyle) - Community Conventions
- [Julia Anti-Patterns](https://jgreener64.github.io/julia-anti-patterns/)

**Real-World Examples:**
- [DataFrames.jl](https://github.com/JuliaData/DataFrames.jl) - Data Manipulation Patterns
- [Flux.jl](https://github.com/FluxML/Flux.jl) - Machine Learning Patterns
- [Julia Base Source](https://github.com/JuliaLang/julia/tree/master/base) - Language Implementation

**SciML & AD Resources:**
- [SciML Style Guide](https://docs.sciml.ai/SciMLStyle/stable/) - Scientific Computing Patterns
- [Lux.jl Documentation](https://lux.csail.mit.edu/) - Explicit State NN for AD
- [ModelingToolkit.jl](https://docs.sciml.ai/ModelingToolkit/) - Symbolic Modeling
- [Zygote.jl](https://fluxml.ai/Zygote.jl/) - Source-to-Source AD
- [ChainRulesCore.jl](https://github.com/JuliaDiff/ChainRulesCore.jl) - Custom Differentiation Rules
- [SciML Tutorials](https://tutorials.sciml.ai/) - Hands-on Learning

---

**Remember**: Idiomatic Julia leverages the language's strengths—multiple dispatch, JIT compilation, and expressive syntax—to write code that is both fast and readable. Type stability and genericity are key to performance, while following community conventions ensures your code integrates seamlessly with the ecosystem.

**Additional Note for SciML developers**: When writing differentiable scientific computing code, **differentiability constraints** take precedence. If a performance pattern conflicts with AD compatibility (e.g., in-place mutation), favor functional/immutable approaches for loss loops, ODE RHS functions, and any code that will be differentiated by Zygote or Enzyme.

---

## REFERENCE IMPLEMENTATION: CORLEONE.JL

The Corleone.jl package (analyzed in `CODEBASE_ANALYSIS.md`) demonstrates **exemplary implementation** of these patterns:

### Key Takeaways from Corleone.jl

1. **Lux layers with type-level flags**: `ControlParameter{T, C, B, SHOOTED, N}` where `SHOOTED` is a Bool at type level
2. **Generated functions for tuple unrolling**: `@generated` used to eliminate loop overhead for fixed-size NamedTuples
3. **Runtime-generated symbolic functions**: `@RuntimeGeneratedFunction` compiles symbolic expressions to native code
4. **Type-stable array reconstruction**: Boolean masks (`keeps`, `replaces`) for selective parameter updates
5. **Deep mapping with Functors**: `Functors.fmapstructure` recursively applies functions to nested parameters
6. **Symbolic indexing integration**: Full `SymbolicIndexingInterface` implementation for timeseries objects
7. **Broadcasting for AD**: Consistently uses `map`, `.` operators, and `Base.Fix1/Fix2` instead of mutation
8. **Time binning**: Partitions large sequences into fixed-size tuples (MAXBINSIZE=100) to prevent stack overflow

### Codebase Assessment

**Corleone.jl achieves A+ grade** in idiomatic Julia patterns:
- ✅ Perfect Lux.jl parameter/state separation
- ✅ Masterful use of generated functions
- ✅ Type-stable operations throughout
- ✅ Full SciMLBase/AD compatibility
- ✅ SymbolicIndexingInterface integration
- ✅ Consistent broadcasting patterns

**See**: `CODEBASE_ANALYSIS.md` for detailed analysis of each pattern with code examples.