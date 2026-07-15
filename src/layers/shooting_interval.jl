"""
$(TYPEDEF)

A single shooting stage in a direct multiple-shooting discretization. It holds the
indices (numeric or symbolic) of the state variables whose initial conditions are
treated as tunable parameters (`variable_id`), an initializer, optional bounds, and
the time span for this stage.

When called with a `DEProblem`, it constructs a new problem whose initial condition
has the free components replaced by the current parameter values while the remaining
components keep the original `u0`.

# Fields
$(FIELDS)
"""
@concrete struct ShootingInterval <: LuxCore.AbstractLuxLayer
    "Tunable initial conditons"
    variable_id
    "Initializer for the variable"
    init
    "Bounds"
    bounds
    "The tspan for the initial condition"
    tspan
end

function ShootingInterval(
        problem,
        variable_id,
        tspan = problem.tspan;
        init = nothing,
        bounds = nothing
    )
    return ShootingInterval(
        variable_id,
        isnothing(init) ? (
                isempty(variable_id) ? eltype(problem.u0)[] : getsym(problem, variable_id)(problem)
            ) : Base.Fix1(init, problem),
        bounds,
        tspan
    )
end


LuxCore.display_name(pc::ShootingInterval) = begin
    x = pc.variable_id
    SymbolicIndexingInterface.hasname(x) && return SymbolicIndexingInterface.getname(x)
    Symbol(x)
end

function LuxCore.initialparameters(rng::Random.AbstractRNG, ic::ShootingInterval)
    return maybecallme(ic.init, rng)
end

LuxCore.initialstates(::Random.AbstractRNG, s::ShootingInterval) = (;
    tspan = s.tspan,
)

function get_lower_bound(pc::ShootingInterval, ps, st)
    (; bounds) = pc
    isnothing(bounds) && return get_lower_bound(ps)
    return first_or_first(bounds, ps, st)
end

function get_upper_bound(pc::ShootingInterval, ps, st)
    (; bounds) = pc
    isnothing(bounds) && return get_lower_bound(ps)
    return last_or_last(bounds, ps, st)
end


function (s::ShootingInterval)(problem::SciMLBase.AbstractDEProblem, ps, st::NamedTuple)
    var_idx = get_variable_index(problem, s) |> collect
    idx = eachindex(problem.u0)
    A = [i == j for i in idx, j in var_idx]
    B = diagm([(i ∉ var_idx) for i in eachindex(idx)])
    new_problem = remake(
        problem,
        tspan = something(st.tspan, problem.tspan),
        u0 = A * ps .+ B * problem.u0,
    )
    return new_problem, st
end

get_vs_index(sys, x) =
if isa(SymbolicIndexingInterface.symbolic_type(x), SymbolicIndexingInterface.NotSymbolic)
    return x
else
    return SymbolicIndexingInterface.variable_index(sys, x)
end

get_vs_index(sys, x::Base.AbstractVecOrTuple) = map(x) do xi
    get_vs_index(sys, xi)
end

function get_variable_index(container::C, pc::ShootingInterval) where {C}
    (; variable_id) = pc
    return get_vs_index(container, variable_id)
end

get_variable_index(::Nothing, pc::ShootingInterval) = begin
    (; variable_id) = pc
    @assert isa(SymbolicIndexingInterface.symbolic_type(variable_id), SymbolicIndexingInterface.NotSymbolic) "Symbolic indices are only valid when providing a symbolic container!"
    variable_id
end
