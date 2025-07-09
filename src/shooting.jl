"""
$(TYPEDEF)
"""
struct ShootingGrid{T}
    "Shooting timepoints"
    timepoints::T

    function ShootingGrid(tpoints::AbstractVector)
        tpoints = unique!(sort(tpoints))
        new{typeof(tpoints)}(tpoints)
    end
end

function (x::ShootingGrid)(sys)
    (; timepoints) = x
    timepoints = [t for t in timepoints]
    vars = ModelingToolkit.unknowns(sys)
    new_parameters = []
    init_equations = Equation[]
    new_equations = Equation[]
    foreach(vars) do v
        collect_shooting_equations!(new_parameters, new_equations,
            init_equations, v, sys, timepoints,)
    end
    init_sys = System(
        new_equations,
        ModelingToolkit.get_iv(sys),
        [],
        new_parameters,
        name=nameof(sys),
        initialization_eqs=init_equations
    )
    extend_system(init_sys, sys)
end

function find_initial_condition(t, timepoints, p)
    idx = findall(t .== timepoints)
    !isempty(idx) && return p[only(idx)]
    @warn "No initial condition for timepoint $(t) found for $(get_shootingparent(p))!"
    return p[end]
end

@register_symbolic find_initial_condition(t::Real, timepoints::AbstractVector, p::AbstractVector)::Real

function collect_control_shooting!(new_ps, eqs, inits, x, sys, timepoints)
    var = Symbolics.unwrap(x)
    varsym = Symbol(iscall(var) ? operation(var) : var)
    ps = ModelingToolkit.getvar(sys, Symbol(varsym, :ᵢ), namespace=false)
    ts = ModelingToolkit.getvar(sys, Symbol(varsym, :ₜ), namespace=false)
    diff_control = all(is_differentialcontrol.(collect(ps)))
    if diff_control
        collect_variable_shooting!(new_ps, eqs, inits, x, sys, timepoints)
    else
        push!(inits, x ~ _expand_ifelse(ModelingToolkit.get_iv(sys), collect(ts), collect(ps); shooting=true))
    end
end

function collect_variable_shooting!(new_ps, eqs, inits, x, sys, timepoints)
    u0s = Symbolics.hasmetadata(x, Symbolics.VariableDefaultValue) ? Symbolics.getdefaultval(x) : 0.0
    u0 = fill(u0s, length(timepoints))
    bounds = fill(getbounds(x), length(timepoints))
    N = length(timepoints)
    varsym = Symbol(operation(x))
    psym = Symbol(varsym, :ₛ)
    tsym = Symbol(varsym, :ₛ, :ₜ)
    if !istunable(x)
        bounds[1] = (u0s, u0s)
    end
    ps = @parameters begin
        ($(psym))[1:N] = u0, [tunable = true, bounds = (first.(bounds), last.(bounds)), shooting_variable = x]
        ($(tsym))[1:N] = timepoints[1:N], [tunable = false, shooting = true]
    end
    push!(inits, x ~ find_initial_condition(ModelingToolkit.get_iv(sys), ps[2], ps[1]))
    append!(new_ps, ps)
end

function collect_cost_shooting!(new_ps, eqs, inits, x, sys, timepoints)
    push!(inits, x ~ zero(Symbolics.symtype(x)))
end

function collect_shooting_equations!(new_ps, eqs, inits, x, sys, timepoints, initializer=nothing)
    if isinput(x)
        collect_control_shooting!(new_ps, eqs, inits, x, sys, timepoints)
    elseif is_costvariable(x)
        collect_cost_shooting!(new_ps, eqs, inits, x, sys, timepoints)
    else
        collect_variable_shooting!(new_ps, eqs, inits, x, sys, timepoints)
    end
    return
end

# We follow this pattern: We take the initialization equations.
# If the lhs is a state, we assume it is a shooting variable
# If the lhs is a state(0) we assume it is the corresponding parameter
function build_shooting_initializer(sys)
    init_eqs = initialization_equations(sys)
    lhs = map(x -> x.lhs, init_eqs)
    vars = unknowns(sys)
    idx = [findfirst(Base.Fix1(isequal, xi), lhs) for xi in vars]
    shooting_equations = map(x -> x.rhs, init_eqs[idx])
    return first(ModelingToolkit.generate_custom_function(sys, shooting_equations, []; expression=Val{false}))
end
