struct ShootingGrid{T}
    "Shooting timepoints"
    timepoints::T
    "Shooting node initialization"
    initializer::AbstractNodeInitialization

    function ShootingGrid(tpoints::AbstractVector, initializer::AbstractNodeInitialization)
        tpoints = unique!(sort(tpoints))
        new{typeof(tpoints)}(tpoints, initializer)
    end
end

function (x::ShootingGrid)(sys)
    (; timepoints, initializer) = x
    t0, tinf = ModelingToolkit.get_tspan(sys)
    # Clamp the shooting points
    timepoints = unique(vcat(t0, [t for t in timepoints if t0 < t < tinf], tinf))
    vars = ModelingToolkit.unknowns(sys)
    dependent_vars, free_vars = filter(!isinput, vars), filter(isinput, vars)
    @info vars dependent_vars free_vars
    new_parameters = []
    initialization_equations = Equation[]
    new_equations = Equation[]
    new_defs = copy(ModelingToolkit.defaults(sys))
    foreach(vars) do v
        collect_shooting_equations!(new_parameters, new_equations,
                initialization_equations, v, sys, timepoints, initializer.init)
        delete!(new_defs, v)
    end

    init_sys = ODESystem(
        new_equations,
        ModelingToolkit.get_iv(sys),
        [],
        new_parameters,
        name=nameof(sys),
        defaults=new_defs,
        initialization_eqs=initialization_equations
    )
    newsys = extend(init_sys, sys)
    @set newsys.tspan = (t0, tinf)
end

function find_initial_condition(t, timepoints, p)
    idx = findall(t .== timepoints)
    !isempty(idx) && return p[only(idx)]
    return p[end]
end

@register_symbolic find_initial_condition(t::Real, timepoints::AbstractVector, p::AbstractVector)::Real

function collect_shooting_equations!(new_ps, eqs, inits, x, sys, timepoints, initializer=nothing)
    tspan = ModelingToolkit.get_tspan(sys)
    # TODO: Check if x is # SymbolicUtils.FnType
    varsym = Symbol(operation(x))
    N = length(unique(vcat(first(tspan), timepoints, last(tspan))))
    fixed = false
    if !ModelingToolkit.istunable(x) && (timepoints[1] == tspan[1] && timepoints[2] == tspan[2])
        fixed = true
        N -= 1
    end
    u0 = Symbolics.hasmetadata(x, Symbolics.VariableDefaultValue) ? Symbolics.getdefaultval(x) : 0.0
    defval = isnothing(initializer) ? [u0 for _ in 1:N] : begin
        varfound = findfirst(var -> isequal(var,x), first.(initializer))
        if !isnothing(varfound)
            initializer[varfound].second
        else
            [u0 for _ in 1:N]
        end
    end
    psym = Symbol(varsym, :ₛ) #Symbol(Char(0x2080)))
    tsym = Symbol(varsym, :ₛ, :ₜ) #  Symbol(Char(0x208c)), Symbol(Char(0x2080)))
    tunable = !(is_costvariable(x) || isinput(x))
    lb, ub = getbounds(x)
    bounds = begin
        _lb, _ub = zeros(N), zeros(N)
        # For dependent variables assume that first shooting node is fixed, for controls
        # that are added as states they can be within their respective bounds.
        # TODO: Adapt here for variable initial conditions
        _lb[:] .=  lb
        _ub[:] .= ub
        _lb[1] = isinput(x) ? lb : u0
        _ub[1] = isinput(x) ? ub : u0
        (_lb, _ub)
    end
    ps = isinput(x) ? begin
        params = parameters(sys)
        psym = params[findfirst(x -> x == Symbol(varsym, "ᵢ"), Symbol.(params))]
        tsym = params[findfirst(x -> x == Symbol(varsym, "ₜ"), Symbol.(params))]
        (psym, tsym)
    end :  @parameters begin
        ($(psym))[1:N] = defval, [tunable = tunable, bounds=bounds]
        ($(tsym))[1:N] = timepoints[1:N], [tunable = false, shooting = true]
    end

    push!(inits, x ~ find_initial_condition(ModelingToolkit.get_iv(sys), ps[2], ps[1]))
    append!(new_ps, ps)
    return
end

# We follow this pattern: We take the initialization equations.
# If the lhs is a state, we assume it is a shooting variable
# If the lhs is a state(0) we assume it is the correpsonding parameter
function build_shooting_initializer(sys)
    init_eqs = initialization_equations(sys)
    lhs = map(x -> x.lhs, init_eqs)
    vars = unknowns(sys)
    var_idx = findall(xi->!any(Base.Fix1(isequal, xi), lhs), vars)
    shooting_idx = findall(xi->any(Base.Fix1(isequal, xi), vars), lhs)
    shooting_equations = map(x -> x.rhs, init_eqs[shooting_idx])
    # We assume that if any variable is not present here, we apply a feedthrough
    for i in var_idx
        push!(shooting_equations, vars[i])
    end
    return first(ModelingToolkit.generate_custom_function(sys, shooting_equations, expression = Val{false}))
end