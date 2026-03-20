# Compatibility Notes for test_augmentation_v2.jl

## Issues with the Old Test

The old `test_augmentation_v2.jl` test was written for a different API design that:

1. **Parameter naming**: Used generic names like `:p₁` instead of control parameter names
2. **Fisher in ODE**: Expected both discrete and continuous Fisher to be integrated as ODE variables
3. **Direct layer augmentation**: Expected `augment_fisher(layer, measurements)` to work without accessing the SymbolicSystem

## New API Design

The new API in `augmentation_v2.jl` takes a different approach:

1. **Parameter naming**: Uses actual control parameter names (e.g., `:k`, `:u`)
2. **Discrete Fisher**: Computed post-hoc via `discrete_fisher_information()` function
3. **Requires SymbolicSystem**: Must use `get_symbolic_equations` → `append_sensitivity!` → `add_observed!` workflow

## Why These Changes?

### Parameter Naming
Using actual parameter names is more intuitive and matches the Corleone.jl API better.

### Discrete Fisher Post-Processing
Computing discrete Fisher contributions during ODE integration requires:
- Callback functions at measurement times
- Hybrid system with discrete jumps
- More complex state management

The post-processing approach is simpler and more flexible:
- Users can specify measurement times after solving
- No need for complex callback logic
- Easier to combine multiple measurement types

### SymbolicSystem Workflow
Exposing the SymbolicSystem gives users more control and transparency:
- Can inspect equations at each augmentation step
- More modular and composable
- Easier to debug and extend

## Migration Path

To update code using the old API:

### Old API
```julia
base_layer = SingleShootingLayer(prob, control; algorithm=Tsit5())
sens_layer = augment_sensitivities(base_layer, [:p₁])
measurements = [
    DiscreteMeasurement(disc_cp) => expr,
    ContinuousMeasurement(cont_cp) => expr
]
fisher_layer = augment_fisher(sens_layer, measurements)
```

### New API (Full Control)
```julia
base_layer = SingleShootingLayer(prob, control; algorithm=Tsit5())

# Extract and augment step by step
sys = get_symbolic_equations(base_layer)
append_sensitivity!(sys, [:k])  # Use actual parameter names
add_observed!(sys, 
    DiscreteMeasurement(disc_cp) => (vars, ps, t) -> vars[1],
    ContinuousMeasurement(cont_cp) => (vars, ps, t) -> vars[1]
)

# Create augmented layer
aug_layer = SingleShootingLayer(sys, base_layer)
oed_layer = OEDLayer(sys, aug_layer)

# Solve and extract Fisher
traj, st = oed_layer(initial_condition, ps, st)
F_cont = fisher_information(oed_layer, traj)
F_disc = discrete_fisher_information(oed_layer, traj, meas_times)
F_total = F_cont + F_disc
```

### New API (Convenience Function)
```julia
base_layer = SingleShootingLayer(prob, control; algorithm=Tsit5())

oed_layer = create_oed_layer(
    base_layer,
    [:k],
    DiscreteMeasurement(disc_cp) => (vars, ps, t) -> vars[1],
    ContinuousMeasurement(cont_cp) => (vars, ps, t) -> vars[1]
)

# Use like normal layer
traj, st = oed_layer(initial_condition, ps, st)
F = fisher_information(oed_layer, traj)
```

## Test Coverage

The new API is thoroughly tested in:
- `test/test_augmentation_v2_new_api.jl` - Core functionality (18 tests)
- `test/test_discrete_fisher.jl` - Discrete Fisher computation (3 tests)
- `test/test_helper_functions.jl` - Convenience functions (3 tests)

The old `test_augmentation_v2.jl` is kept for reference but may not pass with the new API.
