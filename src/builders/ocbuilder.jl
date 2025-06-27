"""
$(TYPEDEF)

Takes in a `System` with defined `cost` and optional `constraints` and `consolidate` together with a [`ShootingGrid`](@ref) and [`AbstractControlFormulation`](@ref)s and build the related `OptimizationProblem`.
"""
struct OCProblemBuilder{S,C,G} <: AbstractBuilder
    "The system"
    system::S
    "The controls"
    controls::C
    "The grid"
    grids::G
    "Substitutions"
    substitutions::Dict
end

function OCProblemBuilder(sys::ModelingToolkit.System, controls::C, grids::G, subs::Dict) where {C<:Tuple,G<:Tuple}
    OCProblemBuilder{typeof(sys),C,G}(
        sys, controls, grids, subs
    )
end

function OCProblemBuilder(sys::ModelingToolkit.System, args...)
    controls = filter(Base.Fix2(isa, AbstractControlFormulation), args)
    grid = filter(Base.Fix2(isa, ShootingGrid), args)
    OCProblemBuilder{typeof(sys),typeof(controls),typeof(grid)}(
        sys, controls, grid, Dict()
    )
end

function (prob::OCProblemBuilder)(; kwargs...)
    # Extend the costs
    prob = expand_lagrange!(prob)
    # Extend the controls
    prob = @set prob.system = (only(prob.grids) ∘ tearing)(foldl(∘, prob.controls, init=identity)(prob.system))
    prob = replace_shooting_variables!(prob)
    prob = append_shooting_constraints!(prob)
    prob = @set prob.system = complete(prob.system; add_initial_parameters=false)
    return prob
end
