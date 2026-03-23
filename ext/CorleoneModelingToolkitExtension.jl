module CorleoneModelingToolkitExtension

using Corleone
using ModelingToolkit
using ModelingToolkit: t_nounits, D_nounits

# Access sub-packages through ModelingToolkit
const Sym = ModelingToolkit.Symbolics
const SU = ModelingToolkit.SymbolicUtils

const unwrap = Sym.unwrap
const wrap = Sym.wrap
const iscall = SU.iscall
const operation = SU.operation
const arguments = SU.arguments
const maketerm = SU.maketerm
const toexpr = SU.Code.toexpr

using SciMLBase
using SciMLStructures

import Corleone: SingleShootingLayer, DynamicOptimizationLayer, remake_system, _to_plain_symbol

# ──────────────────────────────────────────────────────
#  Safe name extraction (handles array symbolics)
# ──────────────────────────────────────────────────────

"""
    _safe_getname(p)

Extract a `Symbol` name from a symbolic parameter, handling both scalar parameters
and array symbolics (e.g., `α[1:1]` from `tunable_parameters`).
"""
function _safe_getname(p)
    p = unwrap(p)
    # For array symbolics like α[1:1], operation is getindex
    if iscall(p) && operation(p) == getindex
        base = arguments(p)[1]
        return ModelingToolkit.getname(unwrap(base))
    end
    return ModelingToolkit.getname(p)
end

# Override _to_plain_symbol for MTK symbolic types
function _to_plain_symbol(s::SU.BasicSymbolic)
    if iscall(s)
        op = operation(s)
        if op isa SU.BasicSymbolic
            # For x(t) → operation is x (a BasicSymbolic), get its name
            return ModelingToolkit.getname(op)
        else
            # For Initial(x(t)) etc., fall back to Symbol string representation
            return Symbol(s)
        end
    end
    return ModelingToolkit.getname(s)
end
# Handle Num wrapper
_to_plain_symbol(s::ModelingToolkit.Symbolics.Num) = _to_plain_symbol(unwrap(s))

# ──────────────────────────────────────────────────────
#  remake_system: just return the MTK System as-is
# ──────────────────────────────────────────────────────

remake_system(sys::ModelingToolkit.System, controls) = sys

# ──────────────────────────────────────────────────────
#  Integral utilities
# ──────────────────────────────────────────────────────

function _is_integral(term)
    term = unwrap(term)
    iscall(term) && operation(term) isa Sym.Integral
end

function _collect_integrals(expr)
    integrals = Any[]
    _collect_integrals!(integrals, unwrap(expr))
    return integrals
end

function _collect_integrals!(acc, term)
    term = unwrap(term)
    if _is_integral(term)
        push!(acc, term)
        return
    end
    if iscall(term)
        for arg in arguments(term)
            _collect_integrals!(acc, arg)
        end
    end
    return
end

function _extract_integral_info(integral_term)
    integral_term = unwrap(integral_term)
    op = operation(integral_term)
    domain = op.domain
    interval = domain.domain
    t0 = interval.left
    tinf = interval.right
    integrand = arguments(integral_term)[1]
    return (; t0=t0, tinf=tinf, integrand=integrand)
end

# ──────────────────────────────────────────────────────
#  Input variable → parameter substitution
# ──────────────────────────────────────────────────────

"""
Replace input variables (marked `[input = true]`) with parameters in the system equations.
Returns `(new_equations, substitution_dict)` where the substitution dict maps
original `var(t)` → new parameter symbol.
"""
function _inputs_to_parameters(sys)
    iv = ModelingToolkit.get_iv(sys)
    inputs = filter(v -> ModelingToolkit.isinput(v), ModelingToolkit.unknowns(sys))
    isempty(inputs) && return ModelingToolkit.equations(sys), Dict()

    subs = Dict()
    for inp in inputs
        name = _safe_getname(inp)
        default = ModelingToolkit.getdefault(unwrap(inp))
        p = only(@parameters $name = default)
        subs[inp] = p
    end

    eqs = ModelingToolkit.equations(sys)
    new_eqs = map(eq -> Sym.substitute(eq, subs), eqs)
    return new_eqs, subs
end

# ──────────────────────────────────────────────────────
#  System introspection helpers
# ──────────────────────────────────────────────────────

function _make_control_from_pair(sys, pair::Pair)
    var_expr, timegrid = pair
    var_sym = unwrap(var_expr)
    name = _safe_getname(var_sym)
    t_vec = collect(Float64, timegrid)

    if ModelingToolkit.hasbounds(var_sym)
        lb_val, ub_val = ModelingToolkit.getbounds(var_sym)
        bounds_fn = t -> (fill(Float64(lb_val), length(t)), fill(Float64(ub_val), length(t)))
    else
        bounds_fn = Corleone.default_bounds
    end

    return ControlParameter(t_vec; name=name, bounds=bounds_fn)
end

function _make_tunable_parameter_controls(sys; exclude_names::Set{Symbol}=Set{Symbol}())
    controls = []
    tunables = ModelingToolkit.tunable_parameters(sys)
    for p in tunables
        p_unwrapped = unwrap(p)
        # Skip auto-generated Initial(...) parameters
        if iscall(p_unwrapped) && operation(p_unwrapped) isa ModelingToolkit.Initial
            continue
        end
        # Skip parameters without defaults
        ModelingToolkit.hasdefault(p_unwrapped) || continue

        name = _safe_getname(p_unwrapped)
        name in exclude_names && continue
        default_val = ModelingToolkit.getdefault(p_unwrapped)
        # Check if this is an array-valued parameter (default is a vector)
        if default_val isa AbstractVector
            if ModelingToolkit.hasbounds(p_unwrapped)
                lb_val, ub_val = ModelingToolkit.getbounds(p_unwrapped)
                bounds_fn = t -> (collect(Float64, lb_val), collect(Float64, ub_val))
            else
                bounds_fn = Corleone.default_bounds
            end
            ctrl = ControlParameter(Float64[];
                name=name,
                controls=(rng, t) -> collect(Float64, default_val),
                bounds=bounds_fn,
            )
            push!(controls, ctrl)
        else
            default_float = Float64(default_val)
            if ModelingToolkit.hasbounds(p_unwrapped)
                lb_val, ub_val = ModelingToolkit.getbounds(p_unwrapped)
                bounds_fn = t -> ([Float64(lb_val)], [Float64(ub_val)])
            else
                bounds_fn = Corleone.default_bounds
            end
            ctrl = ControlParameter(Float64[];
                name=name,
                controls=(rng, t) -> [default_float],
                bounds=bounds_fn,
            )
            push!(controls, ctrl)
        end
    end
    return controls
end

function _make_non_tunable_parameter_controls(sys; exclude_names::Set{Symbol}=Set{Symbol}())
    controls = []
    all_params = ModelingToolkit.parameters(sys)
    tunables = Set(_safe_getname.(filter(p -> !(iscall(unwrap(p)) && operation(unwrap(p)) isa ModelingToolkit.Initial), ModelingToolkit.tunable_parameters(sys))))

    for p in all_params
        p_unwrapped = unwrap(p)
        # Skip auto-generated Initial(...) parameters
        if iscall(p_unwrapped) && operation(p_unwrapped) isa ModelingToolkit.Initial
            continue
        end
        # Skip parameters without defaults
        ModelingToolkit.hasdefault(p_unwrapped) || continue

        name = _safe_getname(p_unwrapped)
        name in tunables && continue
        name in exclude_names && continue

        default_val = ModelingToolkit.getdefault(p_unwrapped)
        if default_val isa AbstractVector
            ctrl = FixedControlParameter(Float64[];
                name=name,
                controls=(rng, t) -> collect(Float64, default_val),
            )
        else
            ctrl = FixedControlParameter(Float64[];
                name=name,
                controls=(rng, t) -> [Float64(default_val)],
            )
        end
        push!(controls, ctrl)
    end
    return controls
end

# ──────────────────────────────────────────────────────
#  tspan derivation from symbolic expressions
# ──────────────────────────────────────────────────────

function _collect_timepoints_from_symbolic(expr)
    timepoints = Float64[]
    _collect_timepoints_from_symbolic!(timepoints, unwrap(expr))
    return timepoints
end

function _collect_timepoints_from_symbolic!(acc, term)
    term = unwrap(term)
    if _is_integral(term)
        info = _extract_integral_info(term)
        push!(acc, Float64(info.t0))
        push!(acc, Float64(info.tinf))
        _collect_timepoints_from_symbolic!(acc, unwrap(info.integrand))
        return
    end
    if iscall(term)
        args = arguments(term)
        if length(args) == 1 && args[1] isa Number
            push!(acc, Float64(args[1]))
        end
        for arg in args
            _collect_timepoints_from_symbolic!(acc, unwrap(arg))
        end
    end
    return
end

function _tspan_from_expressions(cost, constraints)
    timepoints = Float64[]

    for c in cost
        append!(timepoints, _collect_timepoints_from_symbolic(c))
    end

    for con in constraints
        if con isa Sym.Inequality
            append!(timepoints, _collect_timepoints_from_symbolic(con.lhs))
            append!(timepoints, _collect_timepoints_from_symbolic(con.rhs))
        elseif con isa Sym.Equation
            append!(timepoints, _collect_timepoints_from_symbolic(con.lhs))
            append!(timepoints, _collect_timepoints_from_symbolic(con.rhs))
        end
    end

    isempty(timepoints) && error("Cannot derive tspan: no timepoints found in cost or constraints.")
    return (minimum(timepoints), maximum(timepoints))
end

# ──────────────────────────────────────────────────────
#  Symbolic → Expr conversion
# ──────────────────────────────────────────────────────

_sym_to_expr(sym_expr) = toexpr(unwrap(sym_expr))

function _normalize_constraint_to_expr(con)
    if con isa Sym.Inequality
        lhs_expr = _sym_to_expr(con.lhs)
        rhs_expr = _sym_to_expr(con.rhs)
        op = con.relational_op == Sym.geq ? :(>=) : :(<=)
        return Expr(:call, op, lhs_expr, rhs_expr)
    elseif con isa Sym.Equation
        lhs_expr = _sym_to_expr(con.lhs)
        rhs_expr = _sym_to_expr(con.rhs)
        return Expr(:call, :(==), lhs_expr, rhs_expr)
    else
        error("Unknown constraint type: $(typeof(con))")
    end
end

# ──────────────────────────────────────────────────────
#  Integral replacement
# ──────────────────────────────────────────────────────

function _replace_integrals(expr, integral_map, iv)
    expr = unwrap(expr)

    if _is_integral(expr)
        for (integral_term, (qvar, info)) in integral_map
            if isequal(unwrap(integral_term), expr)
                return qvar(info.tinf) - qvar(info.t0)
            end
        end
    end

    if !iscall(expr)
        return wrap(expr)
    end

    op = operation(expr)
    args = arguments(expr)
    new_args = map(a -> unwrap(_replace_integrals(a, integral_map, iv)), args)

    if any(i -> !isequal(unwrap(args[i]), new_args[i]), eachindex(args))
        return wrap(maketerm(typeof(expr), op, new_args, nothing))
    end
    return wrap(expr)
end

# ──────────────────────────────────────────────────────
#  SingleShootingLayer from ModelingToolkit.System
# ──────────────────────────────────────────────────────

function SingleShootingLayer(
    sys::ModelingToolkit.System,
    controls::Pair...;
    algorithm::SciMLBase.AbstractDEAlgorithm,
    tspan::Tuple{<:Real, <:Real},
    kwargs...
)
    iv = ModelingToolkit.get_iv(sys)
    new_eqs, input_subs = _inputs_to_parameters(sys)
    new_sys = ModelingToolkit.System(new_eqs, iv; name=ModelingToolkit.nameof(sys))
    completed_sys = ModelingToolkit.complete(new_sys)
    prob = ODEProblem(completed_sys, Dict(), tspan; check_compatibility=false, kwargs...)

    input_controls = [_make_control_from_pair(sys, pair) for pair in controls]
    input_names = Set(Symbol[Symbol(c.name) for c in input_controls])
    tunable_controls = _make_tunable_parameter_controls(completed_sys; exclude_names=input_names)
    non_tunable_controls = _make_non_tunable_parameter_controls(completed_sys; exclude_names=input_names)
    all_controls = (input_controls..., tunable_controls..., non_tunable_controls...)

    return SingleShootingLayer(prob, all_controls...; algorithm=algorithm, kwargs...)
end

# ──────────────────────────────────────────────────────
#  DynamicOptimizationLayer from ModelingToolkit.System
# ──────────────────────────────────────────────────────

function DynamicOptimizationLayer(
    sys::ModelingToolkit.System,
    cost::AbstractVector,
    constraints...;
    controls::AbstractVector{<:Pair} = Pair[],
    algorithm::SciMLBase.AbstractDEAlgorithm,
    kwargs...
)
    iv = ModelingToolkit.get_iv(sys)
    D = ModelingToolkit.Differential(iv)

    # Step 1: Collect all unique integrals from cost and constraints
    all_integrals = Any[]
    for c in cost
        append!(all_integrals, _collect_integrals(c))
    end
    for con in constraints
        if con isa Sym.Inequality
            append!(all_integrals, _collect_integrals(con.lhs))
            append!(all_integrals, _collect_integrals(con.rhs))
        elseif con isa Sym.Equation
            append!(all_integrals, _collect_integrals(con.lhs))
            append!(all_integrals, _collect_integrals(con.rhs))
        end
    end

    # Deduplicate integrals by integrand
    unique_integrals = Dict{Any, Any}()
    quadrature_eqs = Sym.Equation[]
    quadrature_vars = Any[]
    quad_counter = 0

    for integral in all_integrals
        info = _extract_integral_info(integral)
        found = false
        for (_, (existing_var, existing_info)) in unique_integrals
            if isequal(unwrap(info.integrand), unwrap(existing_info.integrand))
                unique_integrals[integral] = (existing_var, info)
                found = true
                break
            end
        end
        if !found
            quad_counter += 1
            qname = Symbol(:_q, quad_counter)
            qvar = only(@variables $qname(..) = 0.0 [tunable = false])
            push!(quadrature_vars, qvar(iv))
            push!(quadrature_eqs, D(qvar(iv)) ~ info.integrand)
            unique_integrals[integral] = (qvar, info)
        end
    end

    # Step 2: Replace integrals with q(tinf) - q(t0)
    new_cost = map(c -> _replace_integrals(c, unique_integrals, iv), cost)

    new_constraints = map(collect(constraints)) do con
        if con isa Sym.Inequality
            new_lhs = _replace_integrals(con.lhs, unique_integrals, iv)
            new_rhs = _replace_integrals(con.rhs, unique_integrals, iv)
            con.relational_op == Sym.geq ? (new_lhs ≳ new_rhs) : (new_lhs ≲ new_rhs)
        elseif con isa Sym.Equation
            new_lhs = _replace_integrals(con.lhs, unique_integrals, iv)
            new_rhs = _replace_integrals(con.rhs, unique_integrals, iv)
            new_lhs ~ new_rhs
        else
            con
        end
    end

    # Step 3: Augment system with quadrature equations, and replace inputs with parameters
    new_eqs, input_subs = _inputs_to_parameters(sys)
    augmented_eqs = vcat(new_eqs, quadrature_eqs)
    augmented_sys = ModelingToolkit.System(augmented_eqs, iv; name=ModelingToolkit.nameof(sys))

    # Apply input substitution to cost and constraints too
    if !isempty(input_subs)
        new_cost = map(c -> Sym.substitute(c, input_subs), new_cost)
        new_constraints = map(new_constraints) do con
            if con isa Sym.Inequality
                lhs = Sym.substitute(con.lhs, input_subs)
                rhs = Sym.substitute(con.rhs, input_subs)
                con.relational_op == Sym.geq ? (lhs ≳ rhs) : (lhs ≲ rhs)
            elseif con isa Sym.Equation
                Sym.substitute(con.lhs, input_subs) ~ Sym.substitute(con.rhs, input_subs)
            else
                con
            end
        end
    end

    # Derive tspan
    tspan = _tspan_from_expressions(cost, collect(constraints))

    completed_sys = ModelingToolkit.complete(augmented_sys)
    prob = ODEProblem(completed_sys, Dict(), tspan; check_compatibility=false, kwargs...)

    # Determine quadrature indices
    all_unknowns = ModelingToolkit.unknowns(completed_sys)
    quadrature_indices = Int[]
    for qv in quadrature_vars
        for (i, u) in enumerate(all_unknowns)
            if isequal(unwrap(qv), unwrap(u))
                push!(quadrature_indices, i)
                break
            end
        end
    end

    # Build controls
    input_controls = [_make_control_from_pair(sys, pair) for pair in controls]
    input_names = Set(Symbol[Symbol(c.name) for c in input_controls])
    tunable_controls = _make_tunable_parameter_controls(completed_sys; exclude_names=input_names)
    non_tunable_controls = _make_non_tunable_parameter_controls(completed_sys; exclude_names=input_names)
    all_ctrl = (input_controls..., tunable_controls..., non_tunable_controls...)

    # Build SingleShootingLayer
    layer = SingleShootingLayer(prob, all_ctrl...;
        algorithm=algorithm,
        quadrature_indices=quadrature_indices,
    )

    # Step 4: Convert to Expr and call existing constructor
    cost_exprs = [_sym_to_expr(c) for c in new_cost]
    obj_expr = length(cost_exprs) == 1 ? cost_exprs[1] : Expr(:call, :+, cost_exprs...)

    constraint_exprs = [_normalize_constraint_to_expr(con) for con in new_constraints]

    return DynamicOptimizationLayer(layer, obj_expr, constraint_exprs...)
end

# ──────────────────────────────────────────────────────
#  Convenience: extract costs/constraints from system
# ──────────────────────────────────────────────────────

function DynamicOptimizationLayer(
    sys::ModelingToolkit.System,
    controls::Pair...;
    algorithm::SciMLBase.AbstractDEAlgorithm,
    kwargs...
)
    cost = ModelingToolkit.get_costs(sys)
    cons = ModelingToolkit.get_constraints(sys)

    return DynamicOptimizationLayer(
        sys, cost, cons...;
        controls=collect(Pair, controls),
        algorithm=algorithm,
        kwargs...
    )
end

end
