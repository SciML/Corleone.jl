# New API for CorleoneOED based on modular augmentation

"""
    SymbolicSystem

A mutable container for symbolic ODE equations that can be augmented with
sensitivities and measurements for optimal experimental design.

# Fields
## Core System
- `sys::SymbolCache`: Symbol cache from the ODE system
- `vars::Vector{Num}`: State variables
- `parameters::Vector{Num}`: Parameters
- `independent_vars::Num`: Time variable
- `equations::Vector{Num}`: Right-hand side of dx/dt = f(x, p, t)

## Sensitivity Information
- `sensitivity_params::Vector{Symbol}`: Parameters for sensitivity analysis
- `sensitivities::Union{Nothing, Matrix{Num}}`: Sensitivity matrix G = ∂x/∂p
- `sensitivity_equations::Union{Nothing, Vector{Num}}`: dG/dt equations

## Measurement Models
- `discrete_measurements::Vector{Tuple{ControlParameter, Any}}`: Discrete measurements
- `continuous_measurements::Vector{Tuple{ControlParameter, Any}}`: Continuous measurements
- `fisher_discrete_vars::Union{Nothing, Vector{Num}}`: Fisher info variables (discrete)
- `fisher_continuous_vars::Union{Nothing, Vector{Num}}`: Fisher info variables (continuous)
- `fisher_discrete_eqs::Union{Nothing, Vector{Num}}`: Fisher accumulation equations (discrete)
- `fisher_continuous_eqs::Union{Nothing, Vector{Num}}`: Fisher accumulation equations (continuous)

## Original Layer Information
- `original_layer::Any`: Reference to the original SingleShootingLayer
- `original_u0::Vector{Float64}`: Original initial conditions
- `original_p::Vector{Float64}`: Original parameters

# Usage
This struct is typically created by `get_symbolic_equations()` and then modified
in place by `append_sensitivity!()` and `add_observed!()`.

# Example
```julia
# Extract and augment
sys = get_symbolic_equations(layer)
append_sensitivity!(sys, [:k])
add_observed!(sys, ContinuousMeasurement(control) => (vars, ps, t) -> vars[1])

# Create augmented layer
aug_layer = SingleShootingLayer(sys, layer)
oed_layer = OEDLayer(sys, aug_layer)
```
"""
mutable struct SymbolicSystem
    # Core system info
    sys::SymbolCache
    vars::Vector{Num}
    parameters::Vector{Num}
    independent_vars::Num
    equations::Vector{Num}
    
    # Sensitivity-related
    sensitivity_params::Vector{Symbol}  # Which parameters to take sensitivities w.r.t.
    sensitivities::Union{Nothing, Matrix{Num}}  # G matrix (nx x np)
    sensitivity_equations::Union{Nothing, Vector{Num}}
    
    # Measurement-related
    discrete_measurements::Vector{Tuple{ControlParameter, Any}}  # (control, expression)
    continuous_measurements::Vector{Tuple{ControlParameter, Any}}
    discrete_measurement_controls::Vector{ControlParameter}  # Controls for discrete measurements (not in ODE)
    continuous_measurement_controls::Vector{ControlParameter}  # Controls for continuous measurements (in ODE)
    fisher_discrete_vars::Union{Nothing, Vector{Num}}  # Variables for discrete Fisher
    fisher_continuous_vars::Union{Nothing, Vector{Num}}  # Variables for continuous Fisher (flattened)
    fisher_discrete_eqs::Union{Nothing, Vector{Num}}
    fisher_continuous_eqs::Union{Nothing, Vector{Num}}
    
    # Original layer info (for reconstruction)
    original_layer::Any
    original_u0::Vector{Float64}
    original_p::Vector{Float64}
end

"""
    DiscreteMeasurement(control, expression)

Defines a discrete measurement model for OED.

# Arguments
- `control::ControlParameter`: The control parameter defining when measurements are taken
- `expression`: A symbolic expression or function (u, p, t) -> measurement

# Example
```julia
disc_cp = ControlParameter(:y => collect(0.0:0.5:2.0))
measurement = DiscreteMeasurement(disc_cp, (u, p, t) -> u[1]^2)
```
"""
struct DiscreteMeasurement
    control::ControlParameter
    expression::Any
end

"""
    ContinuousMeasurement(control, expression)

Defines a continuous measurement model for OED that integrates Fisher information.

# Arguments
- `control::ControlParameter`: The control parameter for measurement weighting over time
- `expression`: A symbolic expression or function (u, p, t) -> measurement

# Example
```julia
cont_cp = ControlParameter(:y_cont => collect(0.0:0.1:2.0))
measurement = ContinuousMeasurement(cont_cp, (u, p, t) -> p[1] + u[1]^2)
```
"""
struct ContinuousMeasurement
    control::ControlParameter
    expression::Any
end

"""
    get_symbolic_equations(layer::SingleShootingLayer)

Extract symbolic equations from a SingleShootingLayer.

Returns a `SymbolicSystem` containing the symbolic representation of the ODE system,
including state variables, parameters, and differential equations. This system can
then be augmented with sensitivities and measurement models.

# Arguments
- `layer::SingleShootingLayer`: The layer to extract equations from

# Returns
- `SymbolicSystem`: A mutable struct containing the symbolic representation

# Example
```julia
# Create a base layer
function odefn!(du, u, p, t)
    du[1] = -p.k * u[1]
end
prob = ODEProblem(odefn!, [1.0], (0.0, 1.0), (k=0.5,))
k = ControlParameter(0.5, (0.1, 2.0), :k)
layer = SingleShootingLayer(prob, k; algorithm=Tsit5())

# Extract symbolic equations
sys = get_symbolic_equations(layer)
```
"""
function get_symbolic_equations(layer::Corleone.SingleShootingLayer)
    prob = Corleone.get_problem(layer)
    
    # Get the symbol cache from the problem, or create a default one
    sys = prob.f.sys
    if isnothing(sys)
        # Create a default system using Corleone's default_system
        sys = Corleone.default_system(prob, layer.controls)
        # Need to remake the problem with this system
        fnew = ODEFunction(prob.f.f; sys = sys)
        prob = remake(prob, f = fnew)
    end
    
    # Get state symbols and create Num variables
    states = SymbolicIndexingInterface.variable_symbols(sys)
    sort!(states, by = Base.Fix1(SymbolicIndexingInterface.variable_index, sys))
    vars = Symbolics.variable.(states)
    vars = Symbolics.setdefaultval.(vars, vec(prob.u0))
    
    # Get parameter symbols
    ps = SymbolicIndexingInterface.parameter_symbols(sys)
    sort!(ps, by = Base.Fix1(SymbolicIndexingInterface.parameter_index, sys))
    parameters = Symbolics.variable.(ps)
    p0, _ = SciMLStructures.canonicalize(SciMLStructures.Tunable(), prob.p)
    parameters = Symbolics.setdefaultval.(parameters, p0)
    
    # Get independent variable (time)
    ts = SymbolicIndexingInterface.independent_variable_symbols(sys)
    independent_vars = Symbolics.variable(only(ts))
    
    # Symbolically evaluate the equations
    f = prob.f.f
    eqs = if SciMLBase.isinplace(prob)
        out = zero(vars)
        f(out, vars, parameters, independent_vars)
        out
    else
        f(vars, parameters, independent_vars)
    end
    
    return SymbolicSystem(
        sys, vars, parameters, independent_vars, eqs,
        Symbol[], nothing, nothing,  # sensitivity fields
        Tuple{ControlParameter, Any}[], Tuple{ControlParameter, Any}[],  # measurements
        ControlParameter[], ControlParameter[],  # discrete and continuous measurement controls
        nothing, nothing, nothing, nothing,  # Fisher fields
        layer, copy(prob.u0), copy(p0)  # original info
    )
end

"""
    append_sensitivity!(symbolic_system::SymbolicSystem, params::Vector{Symbol})

Append forward sensitivity equations to the symbolic system.

This adds sensitivity equations dG/dt = (∂f/∂x)G + ∂f/∂p where G is the sensitivity
matrix of states with respect to parameters. The sensitivity equations are computed
symbolically using automatic differentiation.

# Arguments
- `symbolic_system::SymbolicSystem`: The system to augment
- `params::Vector{Symbol}`: Parameter symbols to compute sensitivities for

# Modifies
The `symbolic_system` in place by adding sensitivity variables and equations.

# Example
```julia
sys = get_symbolic_equations(layer)
append_sensitivity!(sys, [:k])  # Add sensitivities w.r.t. parameter k
```

# Mathematical Background
For an ODE system dx/dt = f(x, p, t), the sensitivity matrix G = ∂x/∂p satisfies:
    dG/dt = (∂f/∂x)G + ∂f/∂p
where G is an (nx × np) matrix with nx states and np parameters.
"""
function append_sensitivity!(symbolic_system::SymbolicSystem, params::Vector{Symbol})
    (; vars, parameters, equations, independent_vars) = symbolic_system
    
    # Find parameter indices
    param_names = Symbol.(parameters)
    param_indices = [findfirst(==(p), param_names) for p in params]
    if any(isnothing, param_indices)
        missing_params = params[findall(isnothing, param_indices)]
        error("Parameters not found in system: $missing_params")
    end
    
    psubset = parameters[param_indices]
    np = length(psubset)
    nx = length(vars)
    
    # Create sensitivity variables G (nx x np)
    G = Symbolics.variables(:G, 1:nx, 1:np)
    G0 = zeros(eltype(symbolic_system.original_u0), nx, np)
    G = Symbolics.setdefaultval.(G, G0)
    
    # Compute Jacobians
    dfdx = Symbolics.jacobian(equations, vars)
    dfdp = Symbolics.jacobian(equations, psubset)
    
    # Sensitivity equations: dG/dt = df/dx * G + df/dp
    sens_eqs = vec(dfdx * G + dfdp)
    
    # Store in system
    symbolic_system.sensitivity_params = params
    symbolic_system.sensitivities = G
    symbolic_system.sensitivity_equations = sens_eqs
    
    return symbolic_system
end

"""
    append_sensitivity!(symbolic_system::SymbolicSystem)

Append forward sensitivity equations for all parameters in the system.
"""
function append_sensitivity!(symbolic_system::SymbolicSystem)
    params = Symbol.(symbolic_system.parameters)
    return append_sensitivity!(symbolic_system, params)
end

"""
    add_observed!(symbolic_system::SymbolicSystem, measurements...)

Add measurement models to the symbolic system and generate Fisher information equations.

This function adds discrete and/or continuous measurements to the system. For continuous
measurements, it automatically generates Fisher information accumulation equations:
    dF/dt = G_weighted' * G_weighted
where G_weighted = (∂h/∂x)G and h is the measurement function.

# Arguments
- `symbolic_system::SymbolicSystem`: The system to augment (must have sensitivities)
- `measurements...`: Variable number of `DiscreteMeasurement` or `ContinuousMeasurement` objects

# Requirements
- Must call `append_sensitivity!` before `add_observed!`

# Example
```julia
sys = get_symbolic_equations(layer)
append_sensitivity!(sys, [:k])

# Add measurements
disc_meas = DiscreteMeasurement(disc_cp) => (vars, ps, t) -> vars[1]^2
cont_meas = ContinuousMeasurement(cont_cp) => (vars, ps, t) -> vars[1]
add_observed!(sys, disc_meas, cont_meas)
```

# Mathematical Background
For continuous measurements h(x, p, t), the Fisher information matrix F satisfies:
    dF/dt = (∂h/∂x · G)' (∂h/∂x · G)
where G = ∂x/∂p is the sensitivity matrix.
"""
function add_observed!(symbolic_system::SymbolicSystem, measurements...)
    if isnothing(symbolic_system.sensitivities)
        error("Must call append_sensitivity! before add_observed!")
    end
    
    (; vars, parameters, independent_vars, sensitivities) = symbolic_system
    
    # Separate discrete and continuous measurements
    for meas in measurements
        if meas isa DiscreteMeasurement
            # Evaluate expression symbolically with (u, p, t) signature
            expr = if meas.expression isa Function
                meas.expression(vars, parameters, independent_vars)
            else
                meas.expression
            end
            
            # Create a modified control with default value of 1.0 for weights
            ctrl = _ensure_weight_defaults(meas.control)
            push!(symbolic_system.discrete_measurements, (ctrl, expr))
            
            # Track the control parameter for discrete measurements
            if !(ctrl in symbolic_system.discrete_measurement_controls)
                push!(symbolic_system.discrete_measurement_controls, ctrl)
            end
        elseif meas isa ContinuousMeasurement
            # Evaluate expression symbolically with (u, p, t) signature
            expr = if meas.expression isa Function
                meas.expression(vars, parameters, independent_vars)
            else
                meas.expression
            end
            
            # Create a modified control with default value of 1.0 for weights
            ctrl = _ensure_weight_defaults(meas.control)
            push!(symbolic_system.continuous_measurements, (ctrl, expr))
            
            # Track the control parameter for continuous measurements
            if !(ctrl in symbolic_system.continuous_measurement_controls)
                push!(symbolic_system.continuous_measurement_controls, ctrl)
            end
        else
            error("Unknown measurement type: $(typeof(meas))")
        end
    end
    
    # Build Fisher information equations for continuous measurements
    if !isempty(symbolic_system.continuous_measurements)
        _add_continuous_fisher!(symbolic_system)
    end
    
    return symbolic_system
end

# Helper function to ensure weight parameters default to 1.0 instead of 0.0
function _ensure_weight_defaults(ctrl::ControlParameter)
    # If the control uses default_controls (which returns zeros), replace with ones
    if ctrl.controls === Corleone.default_controls
        return ControlParameter(
            ctrl.t;
            name = ctrl.name,
            controls = (rng, t) -> ones(eltype(t), length(t)),
            bounds = ctrl.bounds
        )
    else
        return ctrl
    end
end

function _add_continuous_fisher!(symbolic_system::SymbolicSystem)
    (; vars, parameters, sensitivities) = symbolic_system
    np = size(sensitivities, 2)
    
    # Create Fisher variables (symmetric matrix stored as vector)
    selector = triu(trues(np, np))
    n_fisher_vars = sum(selector)
    
    F = Symbolics.variables(:F, 1:np, 1:np)
    F0 = zeros(eltype(symbolic_system.original_u0), np, np)
    F = Symbolics.setdefaultval.(F, F0)
    F_vec = vec(F[selector])  # Store as vector
    
    # Create or collect weight parameters for continuous measurements
    weight_params = Num[]
    for (ctrl, obs_expr) in symbolic_system.continuous_measurements
        weight_sym = Symbol(ctrl.name)
        
        # Check if parameter already exists
        param_idx = findfirst(p -> Symbol(p) == weight_sym, parameters)
        if isnothing(param_idx)
            # Create new parameter for the weight
            weight_var = Symbolics.variable(weight_sym)
            # Get default value: first value from control's time points
            default_val = if isempty(ctrl.t)
                1.0  # Default weight if no time points
            else
                # Call controls with a vector containing the first time point
                first(ctrl.controls(Random.default_rng(), [ctrl.t[1]]))
            end
            weight_var = Symbolics.setdefaultval(weight_var, default_val)
            push!(parameters, weight_var)
            push!(symbolic_system.original_p, default_val)
        else
            weight_var = parameters[param_idx]
        end
        push!(weight_params, weight_var)
    end
    
    # Build Fisher dynamics for each continuous measurement
    fisher_eqs = Num[]
    for (i, (ctrl, obs_expr)) in enumerate(symbolic_system.continuous_measurements)
        weight_var = weight_params[i]
        
        # Compute Jacobian of observable w.r.t. states
        # Handle scalar and vector observables
        obs_vec = obs_expr isa AbstractVector ? obs_expr : [obs_expr]
        dhdx = Symbolics.jacobian(obs_vec, vars)
        
        # Weighted sensitivity: G_weighted = dh/dx * G
        G_weighted = dhdx * sensitivities
        
        # Fisher rate: dF/dt = weight * G_weighted' * G_weighted
        fisher_rate = weight_var * (G_weighted' * G_weighted)
        append!(fisher_eqs, vec(fisher_rate[selector]))
    end
    
    # Sum contributions if multiple continuous measurements
    if length(symbolic_system.continuous_measurements) == 1
        total_fisher_eqs = fisher_eqs
    else
        # Need to sum the rates
        n_eqs_per_meas = length(fisher_eqs) ÷ length(symbolic_system.continuous_measurements)
        total_fisher_eqs = sum([fisher_eqs[(i-1)*n_eqs_per_meas+1:i*n_eqs_per_meas] 
                                for i in 1:length(symbolic_system.continuous_measurements)])
    end
    
    symbolic_system.fisher_continuous_vars = F_vec
    symbolic_system.fisher_continuous_eqs = total_fisher_eqs
    
    return symbolic_system
end

"""
    SingleShootingLayer(symbolic_system::SymbolicSystem, original_layer::SingleShootingLayer)

Create a new SingleShootingLayer from an augmented symbolic system.

This constructs an ODEProblem with the augmented equations (original states +
sensitivities + Fisher information) and creates a new SingleShootingLayer with
the extended state space. The resulting layer can be solved like any other
Corleone layer.

# Arguments
- `symbolic_system::SymbolicSystem`: The augmented system with sensitivities/measurements
- `original_layer::SingleShootingLayer`: The original layer (for copying controls, algorithm, etc.)

# Returns
- `SingleShootingLayer`: A new layer with augmented state space

# State Space Structure
The augmented state vector has the following structure:
1. Original states (nx variables)
2. Sensitivity matrix elements (nx × np variables, stored as vector)
3. Fisher information matrix elements (np × np upper triangle, stored as vector)

# Example
```julia
sys = get_symbolic_equations(layer)
append_sensitivity!(sys, [:k])
add_observed!(sys, ContinuousMeasurement(control) => (vars, ps, t) -> vars[1])

# Create augmented layer
aug_layer = SingleShootingLayer(sys, layer)
```
"""
function Corleone.SingleShootingLayer(
    symbolic_system::SymbolicSystem,
    original_layer::Corleone.SingleShootingLayer
)
    (; vars, parameters, independent_vars, equations) = symbolic_system
    (; sensitivity_equations, sensitivities) = symbolic_system
    (; fisher_continuous_vars, fisher_continuous_eqs) = symbolic_system
    
    # Build the full state vector and equations
    new_vars = copy(vars)
    new_eqs = copy(equations)
    
    if !isnothing(sensitivity_equations)
        append!(new_vars, vec(sensitivities))
        append!(new_eqs, sensitivity_equations)
    end
    
    if !isnothing(fisher_continuous_eqs)
        append!(new_vars, fisher_continuous_vars)
        append!(new_eqs, fisher_continuous_eqs)
    end
    
    # Build ODEFunction
    IIP = true  # We'll use in-place
    foop, fiip = Symbolics.build_function(
        new_eqs, new_vars, parameters, independent_vars; 
        expression = Val{false}, cse = true
    )
    
    # Get initial values
    u0_new = Symbolics.getdefaultval.(new_vars)
    p0_new = Symbolics.getdefaultval.(parameters)
    
    # Create symbol cache
    defaults = Dict(vcat(Symbol.(new_vars), Symbol.(parameters)) .=> vcat(u0_new, p0_new))
    newsys = SymbolCache(
        Symbol.(new_vars), Symbol.(parameters), 
        Symbol(independent_vars);
        defaults = defaults
    )
    
    # Create ODEFunction and problem
    fnew = ODEFunction(fiip; sys = newsys)
    prob = Corleone.get_problem(original_layer)
    new_prob = remake(prob, f = fnew, u0 = u0_new, p = p0_new)
    
    # Create new layer with original controls plus CONTINUOUS measurement controls only
    # (discrete measurement controls are handled by OEDLayer)
    all_controls = vcat(
        collect(values(original_layer.controls.controls)),
        symbolic_system.continuous_measurement_controls
    )
    return Corleone.SingleShootingLayer(
        new_prob, 
        all_controls...;
        algorithm = original_layer.algorithm,
        name = Symbol(string(original_layer.name) * "_augmented")
    )
end

# Export core types and functions
export SymbolicSystem, DiscreteMeasurement, ContinuousMeasurement
export get_symbolic_equations, append_sensitivity!, add_observed!

# OEDLayer and fisher functions are exported from oed_layer.jl

# Export convenience functions
export augment_sensitivities, augment_fisher, create_oed_layer

"""
    augment_sensitivities(layer::SingleShootingLayer, params::Vector{Symbol})

Convenience function that augments a layer with sensitivity equations.

# Example
```julia
base_layer = SingleShootingLayer(prob, control; algorithm=Tsit5())
sens_layer = augment_sensitivities(base_layer, [:p₁])
```
"""
function augment_sensitivities(layer::Corleone.SingleShootingLayer, params::Vector{Symbol})
    sys = get_symbolic_equations(layer)
    append_sensitivity!(sys, params)
    return Corleone.SingleShootingLayer(sys, layer)
end

"""
    augment_fisher(symbolic_system::SymbolicSystem, measurements...)

Convenience function that augments a symbolic system with Fisher information equations.

# Arguments
- `symbolic_system`: A SymbolicSystem that already has sensitivities
- `measurements...`: Variable number of measurement objects (DiscreteMeasurement or ContinuousMeasurement)

# Example
```julia
sys = get_symbolic_equations(layer)
append_sensitivity!(sys, [:k])
augment_fisher(sys, ContinuousMeasurement(control) => (vars, ps, t) -> vars[1])
```
"""
function augment_fisher(symbolic_system::SymbolicSystem, measurements...)
    add_observed!(symbolic_system, measurements...)
    return symbolic_system
end

"""
    create_oed_layer(base_layer::SingleShootingLayer, params::Vector{Symbol}, measurements...)

Convenience function that creates an OED layer with sensitivities and Fisher information in one call.

This is a high-level helper that combines all the steps:
1. Extract symbolic equations
2. Add sensitivity equations
3. Add measurement models
4. Create augmented layer
5. Wrap in OEDLayer

# Arguments
- `base_layer`: Original SingleShootingLayer
- `params`: Parameters to compute sensitivities for
- `measurements...`: Measurement models (DiscreteMeasurement or ContinuousMeasurement)

# Example
```julia
oed_layer = create_oed_layer(
    base_layer, 
    [:k],
    ContinuousMeasurement(control) => (vars, ps, t) -> vars[1]
)
```
"""
function create_oed_layer(
    base_layer::Corleone.SingleShootingLayer,
    params::Vector{Symbol},
    measurements...
)
    sys = get_symbolic_equations(base_layer)
    append_sensitivity!(sys, params)
    add_observed!(sys, measurements...)
    aug_layer = Corleone.SingleShootingLayer(sys, base_layer)
    return OEDLayer(sys, aug_layer)
end
