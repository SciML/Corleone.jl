"""
$(TYPEDEF)

Takes in a `System` with defined `cost` and optional `constraints` and `consolidate` together with a [`ShootingGrid`](@ref) and [`AbstractControlFormulation`](@ref)s and build the related `OptimizationProblem`.
"""
struct OEDProblemBuilder{C,G} <: AbstractBuilder
    "The system"
    system::ModelingToolkit.AbstractSystem
    "The controls"
    controls::C
    "The grid"
    grids::G
    "OED criterion"
    crit::AbstractOEDCriterion
    "Substitutions"
    substitutions::Dict
end

# This makes it so that the OEDBuilder does not replace the costs with anything
__process_costs(::OEDProblemBuilder) = false 
__process_constraints(::OEDProblemBuilder) = true

function OEDProblemBuilder(sys::ModelingToolkit.System, controls::C,
            grids::G, crit::AbstractOEDCriterion,
            subs::Dict) where {C<:Tuple, G<:Tuple,}
    OEDProblemBuilder{C,G}(
        sys, controls, grids, crit, subs
    )
end

function OEDProblemBuilder(sys::ModelingToolkit.System, args...)
    controls = filter(Base.Fix2(isa, AbstractControlFormulation), args)
    grid = filter(Base.Fix2(isa, ShootingGrid), args)
    crit = only(filter(Base.Fix2(isa, AbstractOEDCriterion), args))
    OEDProblemBuilder{typeof(controls),typeof(grid)}(
        sys, controls, grid, crit,  Dict()
    )
end

function expand_equations!(prob::OEDProblemBuilder)
    (; system) = prob
    t = ModelingToolkit.get_iv(system)
    D = Differential(t)
    filter_p = findall(is_statevar, unknowns(system))
    sts = unknowns(system)[filter_p]
    eqs = equations(system)[filter_p]
    _ff = map(eqs) do eq
        if operation(eq.lhs) == Differential(t)
            eq.rhs
        else
            throw(error("Equation is not in standard form ̇x=f(x)."))
        end
    end
    dfdx = Symbolics.jacobian(_ff, sts)
    pp = reduce(vcat, filter(x -> is_uncertain(x) && !(is_shootingvariable(x) || is_shootingpoint(x) || is_localcontrol(x) || is_tstop(x)), parameters(system)))
    np, nx = length(pp), length(sts)
    dfdp = Symbolics.jacobian(_ff, pp)
    G = reduce(vcat, map(1:np) do j
        reduce(vcat, map(1:nx) do i
            Gsym = Symbol("G", "$i", "$j")
            @variables ($(Gsym)(..) = 0.0, [tunable=false, sensitivities=true])
        end)
    end)

    F = []
    for j= 1:np
        for i = 1:np
            Fsym = Symbol("F", "$i", "$j")
            if i <= j
                _f =  @variables ($(Fsym)(..) = 0.0, [tunable=false, fim=true])
                push!(F, only(_f))
            end
        end
    end
    _G = map(x -> x(t), reshape(G, (nx, np)))
    _F = map(x -> x(t), F)

    dg = dfdp .+ dfdx * _G
    sens_eqs = vec(D.(_G)) .~ vec(dg)

    obs_eqs = map(x -> x.rhs, ModelingToolkit.observed(system))
    nh = length(obs_eqs)
    w = reduce(vcat, map(1:nh) do i
        wsym = Symbol("w", "$i")
        @variables ($(wsym)(..) = 1.0, [input=true, bounds=(0.0,1.0), measurements=true])
    end)

    upper_triangle = triu(trues(np, np))
    F_eq = map(enumerate(obs_eqs)) do (i,h_i)
        hix = Symbolics.jacobian([h_i], sts)
        gram = hix * _G
        w[i](t) * gram' * gram
    end |> sum
    fisher_eqs = vec(D.(_F)) .~ vec(F_eq[upper_triangle])
    new_eqs = reduce(vcat, [sens_eqs, fisher_eqs])

    @parameters ϵ = 1.0 [regularization = true, tunable=true, bounds = (0.,1.)]

    oedsys = System(
        new_eqs,
        t,
        reduce(vcat, [vec(_G), vec(_F), [var(t) for var in w]]), [ϵ],
        name = nameof(system),
        )

    newsys = extend_system(system, oedsys)
    newsys = @set newsys.consolidate = ModelingToolkit.get_consolidate(system)
    newsys = @set newsys.constraints = ModelingToolkit.constraints(system)
    return @set prob.system = newsys
end


function replace_controls!(prob::OEDProblemBuilder)
    (; controls, system) = prob

    controls = only(controls)
    t = ModelingToolkit.get_iv(system)
    obs = ModelingToolkit.observed(system)

    obsvar = map(x -> operation(x.lhs), obs)
    sample_var = operation.(filter(is_measurement, unknowns(system)))
    _specs = map(controls.controls) do spec
        if any(isequal(spec.variable), obsvar)
            idx = findfirst(x -> isequal(x, spec.variable), obsvar)
            return merge(spec, (; bounds=(0,1), variable=sample_var[idx]))
        end
        return spec
    end

    # Don't know how this works better
    new_c = begin
        if typeof(controls) <: DirectControlCallback
            DirectControlCallback(_specs)
        elseif typeof(controls) <: IfElseControl
            IfElseControl(_specs)
        end
    end

    return @set only(prob.controls) = new_c
end

function replace_costs!(prob::OEDProblemBuilder)
    (; system, crit) = prob
    costs = ModelingToolkit.get_costs(system)

    tf = last(crit.tspan)

    regu = ModelingToolkit.getvar(system, :ϵ; namespace=false)

    fim_states = sort(filter(is_fim, unknowns(system)), by=x->string(x))

    fim_states_mayer = map(x->operation(x)(tf), fim_states)
    F_mayer = __symmetric_from_vector(fim_states_mayer, regu)
    new_costs = [crit(F_mayer) + regu]

    sys = @set system.costs = new_costs
    return @set prob.system = sys
end

# TODO THIS IS TYPE PIRACY. WE NEED TO FIX THIS :D
function LinearAlgebra.tr(A::Matrix{Symbolics.SymbolicUtils.BasicSymbolic{Real}})
    n = size(A,1)
    sum([A[i,i] for i=1:n])
end

function LinearAlgebra.det(A::Matrix{Symbolics.SymbolicUtils.BasicSymbolic{Real}})
    n = size(A,1)
    prod([A[i,i] for i=1:n])
end

function replace_constraints!(prob::OEDProblemBuilder)
    (; system) = prob

    constraints = ModelingToolkit.constraints(system)
    ob = map(x -> x.lhs, ModelingToolkit.observed(system))
    w = filter(is_measurement, unknowns(system))

    subs = ob .=> w
    new_cs = reduce(vcat, map(constraints) do c
        idx = findfirst(x -> isequal(only(c.lhs.arguments),x), ob)

        if !isnothing(idx)
            new_c  = substitute([c], subs[idx])
            new_c
        else
            c
        end
    end)

    return @set prob.system.constraints = new_cs
end


function (prob::OEDProblemBuilder)(; kwargs...)
    # Extend system with sensitivities, Fisher, and sampling controls
    prob = expand_equations!(prob)
    prob = replace_controls!(prob)
    prob = replace_constraints!(prob)
    prob = expand_lagrange!(prob)
    # Replace costs by criterion
    prob = replace_costs!(prob)
    # Extend the controls
    prob = @set prob.system = (only(prob.grids) ∘ tearing)(foldl(∘, prob.controls, init=identity)(prob.system))
    prob = replace_shooting_variables!(prob)
    prob = append_shooting_constraints!(prob)
    prob = @set prob.system = complete(prob.system; add_initial_parameters=false)
    return prob
end

