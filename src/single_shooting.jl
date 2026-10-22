"""
$(TYPEDEF)
Defines a callable layer that integrates the `AbstractDEProblem` `problem` using the specified
`algorithm`. Controls are assumed to impact differential equation via its parameters `problem.p`
at the positions indicated via `control_indices` and are itself specified via `controls`.
Moreover, initial conditions `problem.u0` that are degrees of freedom to be optimized can be
specified by their indices via `tunable_ic` along with their upper and lower bounds via `bounds_ic`.

# Fields
$(FIELDS)

Note: The orders of both `controls` and `control_indices`, and `bounds_ic` and `tunable_ic`
are assumed to be identical!
"""
struct SingleShootingLayer{P, A, C, B, PB, SI, PI} <: LuxCore.AbstractLuxLayer
    "The underlying differential equation problem"
    problem::P
    "The algorithm with which `problem` is integrated."
    algorithm::A
    "Indices in parameters of `prob` corresponding to controls"
    control_indices::Vector{Int}
    "The controls"
    controls::C
    "Indices of `prob.u0` which are degrees of freedom"
    tunable_ic::Vector{Int}
    "Bounds on the tunable initial conditions of the problem"
    bounds_ic::B
    "Initialization of u"
    state_initialization::SI
    "Indices of `prob.p` which are degrees of freedom. This is derived from control_indices!"
    tunable_p::Vector{Int}
    "Bounds on the tunable parameters of the problem"
    bounds_p::PB
    "Initialization of p"
    parameter_initialization::PI
    "Indices of differential states that are quadratures, i.e. they do not enter into the right hand side of `problem`"
    quadrature_indices::Vector{Int}
end

function default_u0(
        rng::Random.AbstractRNG, problem::SciMLBase.AbstractDEProblem, tunables, (lb, ub)
    )
    return clamp.(problem.u0[tunables], lb[tunables], ub[tunables])
end

function default_p0(
        rng::Random.AbstractRNG, problem::SciMLBase.AbstractDEProblem, parameters, bounds
    )
    pvec, _ = SciMLStructures.canonicalize(SciMLStructures.Tunable(), problem.p)
    return clamp.(pvec[parameters], bounds...)
end

function Base.show(io::IO, layer::SingleShootingLayer)
    type_color, no_color = SciMLBase.get_colorizers(io)

    print(
        io,
        type_color,
        "SingleShootingLayer",
        no_color,
        " with $(length(layer.controls)) controls.\nUnderlying problem: ",
    )

    return Base.show(io, "text/plain", layer.problem)
end

# For DAE problems the default initialisation (`DefaultInit` -> `CheckInit`) requires the
# initial `du0` to satisfy the algebraic constraints within the solver tolerance, which a
# plain guess such as `du0 = 0` for a genuinely non-consistent state does not. Stiff implicit
# DAE integrators (e.g. `DFBDF`) resolve the algebraic relations during the first steps, so
# we trust the given initial derivative (`NoInit`) and let the solver relax onto the
# constraint manifold.
_dae_init_kwargs(prob::SciMLBase.AbstractDAEProblem) = (initializealg = SciMLBase.NoInit(),)
_dae_init_kwargs(prob) = NamedTuple()

# Integrate the shooting segment `[tspan[1], tspan[2]]` of `problem` from `u0` with the
# parameters `p`. Ordinary problems are solved exactly as before; DAE problems additionally
# receive `initializealg = NoInit()` so that the consistency check does not reject the
# (intentionally non-consistent) initial derivative `du0` during optimization.
function _solve_segment(prob::SciMLBase.AbstractDAEProblem, alg, u0, tspan, p, save_start)
    return solve(
        prob, alg;
        u0, dense = false, save_start, save_end = true,
        tspan, p = p, save_everystep = false,
        initializealg = SciMLBase.NoInit(),
    )
end
function _solve_segment(prob, alg, u0, tspan, p, save_start)
    return solve(
        prob, alg;
        u0, dense = false, save_start, save_end = true,
        tspan, p = p, save_everystep = false,
    )
end

function init_problem(prob, alg)
    return remake_problem(prob, SciMLBase.init(prob, alg; _dae_init_kwargs(prob)...))
end

function remake_problem(prob::ODEProblem, state)
    return remake(prob; u0 = state.u)
end

function remake_problem(prob::DAEProblem, state)
    return remake(prob; u0 = state.u, du0 = state.du)
end

"""
$(SIGNATURES)

Constructs a SingleShootingLayer from an `AbstractDEProblem` and a suitable `AbstractDEAlgorithm`
`alg`.

# Arguments
    - `control_indices` : Vector of indices of `prob.p` that denote controls
    - `controls`: Tuple of `ControlParameter` specifying the controls
    - `tunable_ic`: Vector of indices of `prob.u0` that is tunable, i.e., a degree of freedom
    - `bounds_ic` : Vector of tuples of lower and upper bounds of tunable initial conditions
    - `params`: Optional collection selecting the parameters of `prob` to treat as
      **tunable** degrees of freedom. Each entry is either an integer index into the
      canonicalized tunable parameter vector of `prob.p` (excluding any control slots) or a
      `Symbol` naming a model parameter (resolved via the problem's symbol cache; any
      control slot is a known parameter name and is rejected). Parameters not selected here
      are *fixed* at their value in `prob.p` and are excluded from the layer's optimization
      parameters. `nothing` (the default) keeps every non-control parameter tunable. This is
      how a user fits only a *subset* of the model parameters while pinning the rest.
"""
function SingleShootingLayer(
        prob,
        alg;
        controls = [],
        tunable_ic = Int[],
        bounds_ic = nothing,
        state_initialization = default_u0,
        bounds_p = nothing,
        parameter_initialization = default_p0,
        quadrature_indices = Int[],
        params = nothing,
        kwargs...,
    )
    _prob = init_problem(remake(prob; kwargs...), alg)
    controls = collect(controls)
    control_indices = isempty(controls) ? Int[] : Int.(first.(controls))
    # Keep an empty control set as an empty tuple instead of a `Vector{Any}` (which is what
    # `collect([])` yields), and store the controls as a *tuple* rather than a `Vector`: an
    # abstractly-typed empty `controls` field makes `parameterlength`/`get_bounds` infer
    # `Any` for control-less layers, and splatting a `Vector` into the vararg
    # `collect_local_control_bounds(controls...)` erases the count so a control-carrying
    # layer infers `Any` too. Tuples keep both stable.
    controls = isempty(controls) ? () : Tuple(last.(controls))
    u0 = prob.u0
    p_vec, _... = SciMLStructures.canonicalize(SciMLStructures.Tunable(), prob.p)
    tunable_p = sort!(Int.(setdiff(eachindex(p_vec), control_indices)))
    if !isnothing(params)
        symcache = any(x -> x isa Symbol, collect(params)) ?
            retrieve_symbol_cache(prob, control_indices) : nothing
        tunable_p = _resolve_tunable_p(params, tunable_p, control_indices, symcache)
    end
    p_vec = p_vec[tunable_p]
    ic_bounds = isnothing(bounds_ic) ? (to_val(u0, -Inf), to_val(u0, Inf)) : bounds_ic
    p_bounds = isnothing(bounds_p) ? (to_val(p_vec, -Inf), to_val(p_vec, Inf)) : bounds_p
    quadrature_indices = isempty(quadrature_indices) ? Int[] : Int.(collect(quadrature_indices))

    @assert size(ic_bounds[1]) == size(ic_bounds[2]) == size(u0) "The size of the initial states and its bounds is inconsistent."
    @assert size(p_bounds[1]) == size(p_bounds[2]) == size(p_vec) "The size of the initial parameter vector and its bounds is inconsistent."
    @assert all(checkbounds(Bool, u0, quadrature_index) for quadrature_index in quadrature_indices) "Some quadrature indices are inconsistent with the state dimension"

    return SingleShootingLayer(
        _prob,
        alg,
        control_indices,
        controls,
        Int.(tunable_ic),
        ic_bounds,
        state_initialization,
        Int.(tunable_p),
        p_bounds,
        parameter_initialization,
        quadrature_indices
    )
end

# Restrict the default tunable-parameter set (every non-control parameter) to the subset
# chosen by `params`. Integer entries select positions directly into the canonicalized
# tunable parameter vector of `prob.p`; `Symbol` entries are resolved against the problem's
# symbol cache (which lists the non-control parameters by name) and mapped back to their
# raw positions via `pool`. Entries that are control slots or out of range are rejected.
# `symcache` is `nothing` when `params` contains no symbols.
function _resolve_tunable_p(params, pool::Vector{Int}, control_indices::Vector{Int}, symcache)
    sel = Int[]
    for entry in params
        idx = if entry isa Integer
            Int(entry)
        elseif entry isa Symbol
            pos = SymbolicIndexingInterface.parameter_index(symcache, entry)
            isnothing(pos) && throw(
                ArgumentError(
                    "Unrecognized parameter name `$entry` in `params`; " *
                        "known non-control parameters are $(parameter_symbols(symcache))."
                )
            )
            pool[Int(pos)]
        else
            throw(
                ArgumentError(
                    "`params` entries must be integer parameter indices or `Symbol` names; " *
                        "got an entry of type $(typeof(entry))."
                )
            )
        end
        idx in pool || throw(
            ArgumentError(
                "`params` must be a subset of the non-control parameter indices $pool " *
                    "(got an entry resolving to $idx; control slots $(control_indices) and " *
                    "out-of-range indices are not allowed)."
            )
        )
        push!(sel, idx)
    end
    isempty(sel) && throw(
        ArgumentError(
            "`params` selected no tunable parameters; provide at least one parameter index or name."
        )
    )
    return sort!(unique(sel))
end

"""
    get_problem(layer::SingleShootingLayer)

Return the differential-equation problem integrated by `layer`.
"""
get_problem(layer::SingleShootingLayer) = layer.problem

"""
    get_controls(layer::SingleShootingLayer)

Return `(controls, control_indices)`, where `controls` are the control parameter
objects and `control_indices` identify their positions in the problem parameter vector.
"""
get_controls(layer::SingleShootingLayer) = (layer.controls, layer.control_indices)

"""
    get_tspan(layer::SingleShootingLayer)

Return the time span of the underlying differential-equation problem.
"""
get_tspan(layer::SingleShootingLayer) = layer.problem.tspan

"""
    get_tunable(layer::SingleShootingLayer)

Return the indices of tunable initial-condition entries.
"""
get_tunable(layer::SingleShootingLayer) = layer.tunable_ic
function get_params(layer::SingleShootingLayer)
    return setdiff(eachindex(layer.problem.p), layer.control_indices)
end
function __get_tunable_p(layer::SingleShootingLayer)
    return first(SciMLStructures.canonicalize(SciMLStructures.Tunable(), layer.problem.p))
end
get_quadrature_indices(layer::SingleShootingLayer) = layer.quadrature_indices

function get_bounds(layer::SingleShootingLayer; shooting = false, kwargs...)
    (; bounds_ic, bounds_p, controls, tunable_ic, quadrature_indices) = layer
    state_indices = setdiff(eachindex(first(bounds_ic)), quadrature_indices)
    bounds_ic = shooting ? map(Base.Fix2(getindex, state_indices), bounds_ic) : map(Base.Fix2(getindex, tunable_ic), bounds_ic)
    if !isempty(controls)
        control_lb, control_ub = collect_local_control_bounds(controls...; tspan = layer.problem.tspan, kwargs...)
    else
        control_ub = control_lb = eltype(first(bounds_ic))[]
    end
    return (
        (; u0 = first(bounds_ic), p = first(bounds_p), controls = control_lb),
        (; u0 = last(bounds_ic), p = last(bounds_p), controls = control_ub),
    )
end

function LuxCore.initialparameters(rng::Random.AbstractRNG, layer::SingleShootingLayer)
    return __initialparameters(rng, layer)
end
LuxCore.parameterlength(layer::SingleShootingLayer) = __parameterlength(layer)

function LuxCore.initialstates(rng::Random.AbstractRNG, layer::SingleShootingLayer)
    return __initialstates(rng, layer)
end

function __initialparameters(
        rng::Random.AbstractRNG,
        layer::SingleShootingLayer;
        tspan = layer.problem.tspan,
        u0 = layer.problem.u0,
        shooting_layer = false,
        kwargs...,
    )
    (;
        problem,
        state_initialization,
        parameter_initialization,
        tunable_ic,
        bounds_ic,
        bounds_p,
        tunable_p,
        control_indices,
        quadrature_indices,
    ) = layer
    problem = remake(problem; tspan, u0)
    return (;
        u0 = state_initialization(
            rng, problem, shooting_layer ? setdiff(eachindex(u0), quadrature_indices) : tunable_ic, bounds_ic
        ),
        p = parameter_initialization(rng, problem, tunable_p, bounds_p),
        controls = if isempty(layer.controls)
            eltype(layer.problem.u0)[]
        else
            collect_local_controls(rng, layer.controls...; tspan, kwargs...)
        end,
    )
end

function __parameterlength(
        layer::SingleShootingLayer; tspan = layer.problem.tspan, shooting_layer = false, kwargs...
    )
    N = shooting_layer ? prod(size(layer.problem.u0)) - length(layer.quadrature_indices) : size(layer.tunable_ic, 1)
    N += length(layer.tunable_p)
    if !isempty(layer.controls)
        N += sum(layer.controls) do control
            control_length(control; tspan, kwargs...)
        end
    end
    return N
end

function retrieve_symbol_cache(problem::SciMLBase.AbstractDEProblem, control_indices; control_names = [Symbol(:u, _subscript(u_id)) for u_id in 1:length(control_indices)])
    return retrieve_symbol_cache(problem.f.sys, problem.u0, problem.p, control_indices; control_names = control_names)
end

_subscript(i::Integer) = (i |> digits |> reverse .|> dgt -> Char(0x2080 + dgt)) |> join

function retrieve_symbol_cache(::Nothing, u0, p, control_indices; control_names = [Symbol(:u, _subscript(u_id)) for u_id in 1:length(control_indices)])
    p0, _ = SciMLStructures.canonicalize(SciMLStructures.Tunable(), p)
    state_symbols = [Symbol(:x, _subscript(i)) for i in eachindex(u0)]
    u_id = 0
    p_id = 0
    parameter_symbols = [
        if i ∈ control_indices
            u_id += 1
            control_names[u_id]
        else
            Symbol(:p, _subscript(p_id += 1))
        end for i in eachindex(p0)
    ]
    tsym = [:t]
    return _retrieve_symbol_cache(state_symbols, parameter_symbols, tsym, control_indices)
end

function retrieve_symbol_cache(cache::SymbolCache, u0, p, control_indices; kwargs...)
    psym = parameter_symbols(cache)
    vsym = variable_symbols(cache)
    sort!(psym; by = xi -> SymbolicIndexingInterface.parameter_index(cache, xi))
    sort!(vsym; by = xi -> SymbolicIndexingInterface.variable_index(cache, xi))
    return _retrieve_symbol_cache(
        vsym, psym, independent_variable_symbols(cache), control_indices
    )
end

function _retrieve_symbol_cache(xs, ps, t, idx)
    nonidx = filter(∉(idx), eachindex(ps))
    return SymbolCache(vcat(xs, ps[idx]), ps[nonidx], t)
end

struct InitialConditionRemaker <: Function
    sorting::Vector{Int}
    constants::Vector{Int}
end

function (ic::InitialConditionRemaker)(
        u::AbstractVector{T}, u0::AbstractArray
    ) where {T <: Number}
    (; constants, sorting) = ic
    isempty(u) && return T.(u0)
    return reshape(vcat(vec(u0[constants]), u)[sorting], size(u0))
end

function (ic::InitialConditionRemaker)(::Any, u0::AbstractArray)
    return u0
end

statelength(ic::InitialConditionRemaker) = length(ic.sorting)

function __initialstates(
        rng::Random.AbstractRNG,
        layer::SingleShootingLayer;
        tspan = layer.problem.tspan,
        shooting_layer = false,
        kwargs...,
    )
    (; tunable_ic, control_indices, tunable_p, problem, controls, quadrature_indices) = layer
    (; u0) = problem

    initial_condition = if !shooting_layer
        constant_ics = setdiff(eachindex(u0), tunable_ic)
        sorting = sortperm(vcat(constant_ics, tunable_ic))
        shape = size(u0)
        constants = constant_ics
        InitialConditionRemaker(sorting, constants)
    else
        constant_ic = quadrature_indices
        tunables = setdiff(eachindex(u0), constant_ic)
        sorting = sortperm(vcat(constant_ic, tunables))
        InitialConditionRemaker(sorting, constant_ic)
    end
    # Setup the parameters
    p_vec, repack, _ = SciMLStructures.canonicalize(
        SciMLStructures.Tunable(), layer.problem.p
    )

    # We filter controls which do not act on the dynamics
    active_controls = control_indices .<= lastindex(p_vec)
    control_indices = control_indices[active_controls]
    controls = controls[active_controls]

    # Parameter matrix maps the *tunable* parameters (the `params` subset, or every
    # non-control parameter by default) onto their slots in the full parameter vector;
    # non-tunable, non-control parameters are *fixed* at their value in `prob.p` and enter
    # as a constant offset (`fixed_params`), so they are not degrees of freedom.
    parameter_matrix = zeros(Bool, size(p_vec, 1), length(tunable_p))
    control_matrix = zeros(Bool, size(p_vec, 1), size(control_indices, 1))
    fixed_params = zeros(eltype(p_vec), size(p_vec, 1))
    param_id = 0
    control_id = 0
    for i in eachindex(p_vec)
        if i ∈ control_indices
            control_matrix[i, control_id += 1] = true
        elseif i ∈ tunable_p
            parameter_matrix[i, param_id += 1] = true
        else
            fixed_params[i] = p_vec[i]
        end
    end
    parameter_vector = let repack = repack, A = parameter_matrix, B = control_matrix,
            f = fixed_params
        function (params, controls)
            return repack(A * params .+ B * controls .+ f)
        end
    end

    # Next we setup the tspans and the indices
    if !isempty(controls)
        grid = build_index_grid(controls...; tspan, subdivide = 100)
        tspans = collect_tspans(controls...; tspan, subdivide = 100)
    else
        grid = Int[i for i in control_indices]
        tspans = (tspan,)
    end
    shooting_indices = zeros(Bool, size(u0, 1) + length(controls))
    if shooting_layer
        shooting_indices[setdiff(eachindex(u0), quadrature_indices)] .= true
        for (i, c) in enumerate(controls)
            shooting_indices[lastindex(u0) + i] = !find_shooting_indices(first(tspans), c)
        end
    end
    shooting_indices = findall(shooting_indices)
    control_names = [controls[i].name for i in sortperm(control_indices)]
    symcache = retrieve_symbol_cache(problem, control_indices; control_names)
    return (;
        initial_condition,
        index_grid = grid,
        tspans,
        parameter_vector,
        symcache,
        shooting_indices,
        active_controls = find_active_controls(grid),
    )
end

find_active_controls(grid::AbstractArray) = map(unique, eachrow(grid))
find_active_controls(grid::Tuple) = unique!(reduce(vcat, map(find_active_controls, grid)))

function (layer::SingleShootingLayer)(::Any, ps, st, shooting_layer = false)
    (; initial_condition) = st
    fixval = shooting_layer ? zeros(statelength(initial_condition)) : layer.problem.u0
    u0 = initial_condition(ps.u0, fixval)
    return layer(u0, ps, st)
end

function (layer::SingleShootingLayer)(u0::AbstractArray, ps, st)
    (; problem, algorithm) = layer
    (; p, controls) = ps
    (; index_grid, tspans, parameter_vector, symcache) = st
    params = Base.Fix1(parameter_vector, p)
    # Returns the states as DiffEqArray
    solutions = sequential_solve(
        problem, algorithm, u0, params, controls, index_grid, tspans, symcache
    )
    return solutions, st
end

function build_optimal_control_solution(u, t, p, sys)
    return Trajectory(sys, u, p, t, empty(u), Int[])
end

sequential_solve(args...) = _sequential_solve(args...)

# Slicing can return a different container than `u0`: under ReverseDiff >= 1.17.2 the slice is a
# `TrackedArray` (the controls are tracked views), which an in-place solve cannot write into.
# For mutable `u0`, copy the elements into a container like `u0` instead; immutable containers
# (e.g. `SVector`, `SizedVector`) keep the slice unchanged.
function _carry_state(u0, x)
    y = x[eachindex(u0)]
    (y isa typeof(u0) || !ismutable(u0)) && return y
    z = similar(u0, eltype(y))
    for (i, j) in zip(eachindex(z), eachindex(y))
        z[i] = y[j]
    end
    return z
end

@generated function _sequential_solve(
        problem, alg, u0, param, ps, indexgrids::NTuple{N}, tspans::NTuple{N, Tuple}, sys
    ) where {N}
    solutions = [gensym() for _ in 1:N]
    u0s = [gensym() for _ in 1:N]
    ex = Expr[]
    u_ret_expr = :(vcat())
    t_ret_expr = :(vcat())
    push!(ex, :($(u0s[1]) = u0))

    for i in 1:N
        push!(
            ex,
            :(
                $(solutions[i]) = _sequential_solve(
                    problem, alg, $(u0s[i]), param, ps, indexgrids[$(i)], tspans[$(i)], sys
                )
            ),
        )
        if i < N
            push!(u_ret_expr.args, :($(solutions[i]).u[1:(end - 1)]))
            push!(t_ret_expr.args, :($(solutions[i]).t[1:(end - 1)]))
            push!(ex, :($(u0s[i + 1]) = _carry_state(u0, last($(solutions[i]).u))))
        else
            push!(u_ret_expr.args, :($(solutions[i]).u))
            push!(t_ret_expr.args, :($(solutions[i]).t))
        end
    end
    push!(
        ex,
        :(
            return build_optimal_control_solution(
                $(u_ret_expr), $(t_ret_expr), param.x, sys
            )
        ), # Was kommt hier raus
    )
    return Expr(:block, ex...)
end

@generated function _sequential_solve(
        problem,
        alg,
        u0,
        param,
        ps,
        index_grid::AbstractArray,
        tspans::NTuple{N, Tuple{<:Real, <:Real}},
        sys,
    ) where {N}
    solutions = [gensym() for _ in 1:N]
    u0s = [gensym() for _ in 1:N]
    ex = Expr[]
    u_ret_expr = :(vcat())
    t_ret_expr = :(vcat())
    psym = [gensym() for _ in 1:N]
    push!(ex, :($(u0s[1]) = u0))
    for i in 1:N
        push!(ex, :($(psym[i]) = getindex(ps, index_grid[:, $(i)])))
        push!(
            ex,
            :(
                $(solutions[i]) = _solve_segment(
                    problem, alg, $(u0s[i]), tspans[$(i)], param($(psym[i])), $(i == 1)
                )
            ),
        )
        push!(u_ret_expr.args, :(Base.Fix2(vcat, $(psym[i])).($(solutions[i]).u)))
        push!(t_ret_expr.args, :($(solutions[i]).t))
        if i < N
            push!(ex, :($(u0s[i + 1]) = $(solutions[i]).u[end]))
        end
    end
    push!(
        ex,
        :(
            return build_optimal_control_solution(
                $(u_ret_expr), $(t_ret_expr), param.x, sys
            )
        ), # Was kommt hier raus
    )
    return Expr(:block, ex...)
end

function _parallel_solve(::Any, layer::SingleShootingLayer, u0, ps, st)
    @warn "Falling back to using `EnsembleSerial`" maxlog = 1
    return _parallel_solve(EnsembleSerial(), layer, u0, ps, st)
end

__getidx(x, id) = x[id]
__getidx(x::NamedTuple, id) = getproperty(x, id)

function _parallel_solve(
        alg::SciMLBase.EnsembleAlgorithm,
        layer::SingleShootingLayer,
        u0,
        ps,
        st::NamedTuple{fields},
    ) where {fields}
    args = collect(
        ntuple(
            i -> (u0, __getidx(ps, fields[i]), __getidx(st, fields[i])), length(st)
        )
    )
    return mythreadmap(alg, Base.Splat(layer), args)
end

"""
$(SIGNATURES)

Compute the block structure of the hessian of the Lagrangian of an optimal control problem.
As this is a `SingleShootingLayer`, this hessian is dense. See also
[`MultipleShootingLayer`](@ref `Corleone.MultipleShootingLayer`).
"""
function get_block_structure(
        layer::SingleShootingLayer, tspan = layer.problem.tspan, kwargs...
    )
    return vcat(0, LuxCore.parameterlength(layer; tspan, kwargs...))
end
