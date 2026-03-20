"""
    OEDLayer

A Lux container layer for Optimal Experimental Design that wraps an augmented 
SingleShootingLayer and provides Fisher information computation.

This layer is fully differentiable and caches symbolic getters for efficiency.

# Fields
- `layer::L`: The augmented SingleShootingLayer containing ODE + sensitivities + continuous Fisher
- `symbolic_system::S`: The SymbolicSystem metadata
- `discrete_controls::D`: ControlParameters for discrete measurement weights
- `continuous_fisher_getter::C`: Cached getter for continuous Fisher matrix
- `discrete_fisher_getters::Vector{DG}`: Cached getters for discrete Fisher contributions

# Example
```julia
sys = get_symbolic_equations(layer)
append_sensitivity!(sys, [:k])
add_observed!(sys, 
    DiscreteMeasurement(ctrl1, (u,p,t) -> u[1]),
    ContinuousMeasurement(ctrl2, (u,p,t) -> u[1])
)
aug_layer = SingleShootingLayer(sys, layer)
oed_layer = OEDLayer(sys, aug_layer)

ps, st = LuxCore.setup(rng, oed_layer)
(fisher, traj), st = oed_layer(nothing, ps, st)
```
"""
struct OEDLayer{L, S, D, C, DG} <: LuxCore.AbstractLuxContainerLayer{(:layer, :discrete_controls)}
    layer::L
    symbolic_system::S
    discrete_controls::D
    continuous_fisher_getter::C
    discrete_fisher_getters::Vector{DG}
end

"""
    OEDLayer(symbolic_system::SymbolicSystem, layer::SingleShootingLayer)

Construct an OED layer with cached symbolic getters for efficient Fisher computation.
"""
function OEDLayer(symbolic_system::SymbolicSystem, layer::Corleone.SingleShootingLayer)
    # Create ControlParameters container for discrete measurements
    discrete_controls = if isempty(symbolic_system.discrete_measurement_controls)
        nothing
    else
        ctrl_names = Tuple(Symbol(ctrl.name) for ctrl in symbolic_system.discrete_measurement_controls)
        ctrl_values = Tuple(symbolic_system.discrete_measurement_controls)
        discrete_controls_nt = NamedTuple{ctrl_names}(ctrl_values)
        Corleone.ControlParameters(discrete_controls_nt)
    end
    
    # Cache continuous Fisher getter
    continuous_fisher_getter = _build_continuous_fisher_getter(symbolic_system, layer)
    
    # Cache discrete Fisher getters (one per measurement)
    discrete_fisher_getters = _build_discrete_fisher_getters(symbolic_system, layer)
    
    return OEDLayer{
        typeof(layer), 
        typeof(symbolic_system), 
        typeof(discrete_controls),
        typeof(continuous_fisher_getter),
        eltype(discrete_fisher_getters)
    }(
        layer, 
        symbolic_system, 
        discrete_controls,
        continuous_fisher_getter,
        discrete_fisher_getters
    )
end

"""
Build a cached getter function for the continuous Fisher information matrix.
Returns a function that extracts the symmetric Fisher matrix from a trajectory.
"""
function _build_continuous_fisher_getter(symbolic_system::SymbolicSystem, layer::Corleone.SingleShootingLayer)
    if isnothing(symbolic_system.fisher_continuous_vars)
        return nothing
    end
    
    np = length(symbolic_system.sensitivity_params)
    F_vec_syms = symbolic_system.fisher_continuous_vars
    
    # Build symbolic expression to reconstruct symmetric matrix from upper triangular storage
    F_matrix = zeros(Num, np, np)
    idx = 1
    for i in 1:np
        for j in i:np
            F_matrix[i, j] = F_vec_syms[idx]
            if i != j
                F_matrix[j, i] = F_vec_syms[idx]  # Symmetry
            end
            idx += 1
        end
    end
    
    # Convert to expression and create getter
    F_expr = SymbolicUtils.Code.toexpr(F_matrix)
    sys = Corleone.get_problem(layer).f.sys
    fisher_getter = SymbolicIndexingInterface.getsym(sys, F_expr)
    
    return fisher_getter
end

"""
Build cached getter functions for discrete Fisher information contributions.
Returns a vector of (getter_func, ctrl) tuples, one per discrete measurement.
"""
function _build_discrete_fisher_getters(symbolic_system::SymbolicSystem, layer::Corleone.SingleShootingLayer)
    if isempty(symbolic_system.discrete_measurements)
        return []
    end
    
    sys = Corleone.get_problem(layer).f.sys
    np = length(symbolic_system.sensitivity_params)
    vars = symbolic_system.vars
    G_vars = symbolic_system.sensitivities  # nx × np
    
    getters = []
    
    for (ctrl, obs_expr) in symbolic_system.discrete_measurements
        # Compute symbolic unweighted Fisher contribution: (∂h/∂x * G)' * (∂h/∂x * G)
        obs_vec = obs_expr isa AbstractVector ? obs_expr : [obs_expr]
        dhdx = Symbolics.jacobian(obs_vec, vars)  # ny × nx
        
        # Weighted sensitivity: ∂h/∂x * G (ny × np)
        G_weighted = dhdx * G_vars
        
        # Unweighted Fisher contribution: G_weighted' * G_weighted (np × np)
        F_contrib_unweighted = G_weighted' * G_weighted
        
        # Create getter
        F_contrib_expr = SymbolicUtils.Code.toexpr(F_contrib_unweighted)
        F_contrib_getter = SymbolicIndexingInterface.getsym(sys, F_contrib_expr)
        
        push!(getters, (getter=F_contrib_getter, control=ctrl))
    end
    
    return getters
end

"""
    (oed::OEDLayer)(x, ps, st)

Forward pass: compute trajectory and Fisher information matrix.

Returns `((fisher, trajectory), state)` where fisher is the sum of continuous 
and discrete contributions.
"""
function (oed::OEDLayer)(x, ps, st)
    # Call underlying layer to get trajectory
    traj, st_layer = oed.layer(x, ps.layer, st.layer)
    
    # Compute continuous Fisher (from ODE integration)
    fisher_continuous = _compute_continuous_fisher(oed, traj)
    
    # Compute discrete Fisher (from discrete measurements)
    fisher_discrete = _compute_discrete_fisher(oed, traj, ps, st)
    
    # Sum contributions (fully differentiable)
    fisher = fisher_continuous + fisher_discrete
    
    # Update state
    st_new = if isnothing(oed.discrete_controls)
        (layer=st_layer,)
    else
        (layer=st_layer, discrete_controls=st.discrete_controls)
    end
    
    return (fisher, traj), st_new
end

"""
Compute continuous Fisher information using cached getter.
Simply extracts the final value from the trajectory.
"""
function _compute_continuous_fisher(oed::OEDLayer, traj::Trajectory)
    if isnothing(oed.continuous_fisher_getter)
        np = length(oed.symbolic_system.sensitivity_params)
        return zeros(np, np)
    end
    
    # Use cached getter to extract Fisher trajectory
    F_traj = oed.continuous_fisher_getter(traj)
    
    # Return final value
    return F_traj[end]
end

"""
Compute discrete Fisher information using cached getters and weight parameters.
Uses sum() for full differentiability (no mutation).
"""
function _compute_discrete_fisher(oed::OEDLayer, traj::Trajectory, ps, st)
    if isempty(oed.discrete_fisher_getters)
        np = length(oed.symbolic_system.sensitivity_params)
        return zeros(np, np)
    end
    
    # Sum contributions from all discrete measurements (fully differentiable)
    fisher_discrete = sum(oed.discrete_fisher_getters) do getter_info
        (; getter, control) = getter_info
        weight_sym = Symbol(control.name)
        
        # Get unweighted Fisher trajectory
        F_unweighted_traj = getter(traj)
        
        # Get weight trajectory from parameters
        weight_vec = if !isnothing(ps) && haskey(ps, :discrete_controls) && haskey(ps.discrete_controls, weight_sym)
            ps.discrete_controls[weight_sym]
        else
            # Default to ones
            ones(length(control.t))
        end
        
        # Get control time grid and trajectory time grid
        ctrl_times = collect(control.t)
        traj_times = traj.t
        
        # Map weights to trajectory times and compute weighted sum
        # This is fully differentiable using sum()
        sum(enumerate(traj_times)) do (idx, t_traj)
            # Find corresponding weight by finding nearest control time
            widx = _find_nearest_index(ctrl_times, t_traj)
            weight = widx <= length(weight_vec) ? weight_vec[widx] : one(eltype(weight_vec))
            
            # Weight contribution at this time
            weight * F_unweighted_traj[idx]
        end
    end
    
    return fisher_discrete
end

"""
Find index of nearest time point in a sorted time array.
This is differentiable as it doesn't use conditionals that break AD.
"""
function _find_nearest_index(times::AbstractVector, t::Real)
    idx = searchsortedfirst(times, t)
    if idx > length(times)
        return length(times)
    elseif idx == 1
        return 1
    else
        # Return closest of two neighbors
        if abs(times[idx] - t) < abs(times[idx-1] - t)
            return idx
        else
            return idx - 1
        end
    end
end

"""
    fisher_information(oed::OEDLayer, traj::Trajectory)

Extract the final Fisher information matrix from a trajectory.

Uses the cached continuous Fisher getter for efficiency.

# Arguments
- `oed::OEDLayer`: The OED layer wrapper
- `traj::Trajectory`: The solved trajectory

# Returns
- `Matrix{Float64}`: The (np × np) symmetric Fisher information matrix at final time

# Example
```julia
(fisher, traj), st = oed_layer(initial_condition, ps, st)
# fisher is directly returned, or extract separately:
F = fisher_information(oed_layer, traj)
```

# Mathematical Background
The Fisher information matrix measures the curvature of the log-likelihood:
    F_ij = E[(∂log L/∂p_i)(∂log L/∂p_j)]
For Gaussian measurement noise, this becomes:
    F = ∫ G(t)'(∂h/∂x)'(∂h/∂x)G(t) dt
where h is the measurement function and G = ∂x/∂p is the sensitivity matrix.
"""
function fisher_information(oed::OEDLayer, traj::Trajectory)
    return _compute_continuous_fisher(oed, traj)
end

"""
    discrete_fisher_information(oed::OEDLayer, traj::Trajectory, ps)

Compute discrete Fisher information contributions.

Uses cached getters and weight parameters for efficient, differentiable computation.

# Arguments
- `oed::OEDLayer`: The OED layer wrapper
- `traj::Trajectory`: The solved trajectory
- `ps`: Parameter structure containing discrete control weights

# Returns
- `Matrix{Float64}`: The (np × np) discrete Fisher information contribution

# Example
```julia
(fisher, traj), st = oed_layer(nothing, ps, st)
# Discrete Fisher is automatically included in fisher

# Or compute separately
F_disc = discrete_fisher_information(oed_layer, traj, ps)
```

# Note
This function evaluates the measurement Jacobian over the full trajectory and forms
a weighted sum using the control parameters. It's fully differentiable.
"""
function discrete_fisher_information(oed::OEDLayer, traj::Trajectory, ps)
    # Create dummy state for consistency
    st = LuxCore.initialstates(Random.default_rng(), oed)
    return _compute_discrete_fisher(oed, traj, ps, st)
end

"""
    sensitivities(oed::OEDLayer, traj::Trajectory)

Extract sensitivity trajectories from a solution.

Returns the time evolution of the sensitivity matrix G(t) = ∂x(t)/∂p, which
describes how the system states change with respect to parameter perturbations.

# Arguments
- `oed::OEDLayer`: The OED layer wrapper
- `traj::Trajectory`: The solved trajectory

# Returns
- `Vector{Matrix{Float64}}`: Time series of (nx × np) sensitivity matrices

# Example
```julia
(fisher, traj), st = oed_layer(initial_condition, ps, st)
G_traj = sensitivities(oed_layer, traj)
# G_traj[i] is the sensitivity matrix at time traj.t[i]
```
"""
function sensitivities(oed::OEDLayer, traj::Trajectory)
    (; symbolic_system) = oed
    
    if isnothing(symbolic_system.sensitivities)
        error("No sensitivities computed. Did you call append_sensitivity!?")
    end
    
    nx = length(symbolic_system.vars)
    np = length(symbolic_system.sensitivity_params)
    
    # Extract sensitivity variables from trajectory
    sens_start_idx = nx + 1
    sens_end_idx = nx + nx * np
    
    # Map trajectory to sensitivity matrices
    return map(traj.u) do u
        sens_vec = u[sens_start_idx:sens_end_idx]
        reshape(sens_vec, nx, np)
    end
end

# Export
export OEDLayer
export fisher_information, discrete_fisher_information, sensitivities
