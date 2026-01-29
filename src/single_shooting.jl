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
    control_indices::Vector{Int64}
    "The controls"
    controls::C
    "Indices of `prob.u0` which are degrees of freedom"
    tunable_ic::Vector{Int64}
    "Bounds on the tunable initial conditions of the problem"
    bounds_ic::B
    "Initialization of u"
    state_initialization::SI
    "Indices of `prob.p` which are degrees of freedom. This is derived from control_indices!"
    tunable_p::Vector{Int64}
    "Bounds on the tunable parameters of the problem"
    bounds_p::PB
    "Initialization of p"
    parameter_initialization::PI
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

function init_problem(prob, alg)
    return remake_problem(prob, SciMLBase.init(prob, alg))
end

function remake_problem(prob::ODEProblem, state)
    return remake(prob; u0 = state.u)
end

function remake_problem(prob::DAEProblem, state)
    return remake(prob; u0 = state.u, du0 = state.du)
end

"""
$(METHODLIST)
Constructs a SingleShootingLayer from an `AbstractDEProblem` and a suitable inegration method
`alg`.

# Arguments
    - `control_indices` : Vector of indices of `prob.p` that denote controls
    - `controls`: Tuple of `ControlParameter` specifying the controls
    - `tunable_ic`: Vector of indices of `prob.u0` that is tunable, i.e., a degree of freedom
    - `bounds_ic` : Vector of tuples of lower and upper bounds of tunable initial conditions
"""
function SingleShootingLayer(
        prob,
        alg;
        controls = [],
        tunable_ic = Int64[],
        bounds_ic = nothing,
        state_initialization = default_u0,
        bounds_p = nothing,
        parameter_initialization = default_p0,
        kwargs...,
    )
    _prob = init_problem(remake(prob; kwargs...), alg)
    controls = collect(controls)
    control_indices = isempty(controls) ? Int64[] : first.(controls)
    controls = isempty(controls) ? controls : last.(controls)
    u0 = prob.u0
    p_vec, _... = SciMLStructures.canonicalize(SciMLStructures.Tunable(), prob.p)
    tunable_p = setdiff(eachindex(p_vec), control_indices)
    p_vec = p_vec[tunable_p]
    ic_bounds = isnothing(bounds_ic) ? (to_val(u0, -Inf), to_val(u0, Inf)) : bounds_ic
    p_bounds = isnothing(bounds_p) ? (to_val(p_vec, -Inf), to_val(p_vec, Inf)) : bounds_p

    @assert size(ic_bounds[1]) == size(ic_bounds[2]) == size(u0) "The size of the initial states and its bounds is inconsistent."
    @assert size(p_bounds[1]) == size(p_bounds[2]) == size(p_vec) "The size of the initial parameter vector and its bounds is inconsistent."

    return SingleShootingLayer(
        _prob,
        alg,
        control_indices,
        controls,
        tunable_ic,
        ic_bounds,
        state_initialization,
        tunable_p,
        p_bounds,
        parameter_initialization,
    )
end

get_problem(layer::SingleShootingLayer) = layer.problem
get_controls(layer::SingleShootingLayer) = (layer.controls, layer.control_indices)
get_tspan(layer::SingleShootingLayer) = layer.problem.tspan
get_tunable(layer::SingleShootingLayer) = layer.tunable_ic
function get_params(layer::SingleShootingLayer)
    return setdiff(eachindex(layer.problem.p), layer.control_indices)
end
function __get_tunable_p(layer::SingleShootingLayer)
    return first(SciMLStructures.canonicalize(SciMLStructures.Tunable(), layer.problem.p))
end

function get_bounds(layer::SingleShootingLayer; shooting = false, kwargs...)
    (; bounds_ic, bounds_p, controls, tunable_ic) = layer
    bounds_ic = shooting ? bounds_ic : map(Base.Fix2(getindex, tunable_ic), bounds_ic)
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
    ) = layer
    problem = remake(problem; tspan, u0)
    return (;
        u0 = state_initialization(
            rng, problem, shooting_layer ? eachindex(u0) : tunable_ic, bounds_ic
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
    p_vec, _... = SciMLStructures.canonicalize(SciMLStructures.Tunable(), layer.problem.p)
    N = shooting_layer ? prod(size(layer.problem.u0)) : size(layer.tunable_ic, 1)
    N += sum([i ∉ layer.control_indices for i in eachindex(p_vec)])
    if !isempty(layer.controls)
        N += sum(layer.controls) do control
            control_length(control; tspan, kwargs...)
        end
    end
    return N
end

function retrieve_symbol_cache(problem::SciMLBase.DEProblem, control_indices, controls)
    return retrieve_symbol_cache(problem.f.sys, problem.u0, problem.p, control_indices, controls)
end

_subscript(i::Integer) = (i |> digits |> reverse .|> dgt -> Char(0x2080 + dgt)) |> join

function retrieve_symbol_cache(::Nothing, u0, p, control_indices, controls)
    p0, _ = SciMLStructures.canonicalize(SciMLStructures.Tunable(), p)
    state_symbols = [Symbol(:x, _subscript(i)) for i in eachindex(u0)]
    u_id = 0
    p_id = 0
    parameter_symbols = [
        if i ∈ control_indices
            u_id +=1
            controls[u_id].name
        else
            Symbol(:p, _subscript(p_id += 1))
        end for i in eachindex(p0)
    ]
    tsym = [:t]
    return _retrieve_symbol_cache(state_symbols, parameter_symbols, tsym, control_indices)
end

function retrieve_symbol_cache(cache::SymbolCache, u0, p, control_indices)
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
    sorting::Vector{Int64}
    constants::Vector{Int64}
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

function __initialstates(
        rng::Random.AbstractRNG,
        layer::SingleShootingLayer;
        tspan = layer.problem.tspan,
        shooting_layer = false,
        kwargs...,
    )
    (; tunable_ic, control_indices, problem, controls) = layer
    (; u0) = problem

    initial_condition = if !shooting_layer
        constant_ics = setdiff(eachindex(u0), tunable_ic)
        sorting = sortperm(vcat(constant_ics, tunable_ic))
        shape = size(u0)
        constants = constant_ics
        InitialConditionRemaker(sorting, constants)
    else
        constant_ics = Int64[]
        tunable_ic = eachindex(u0)
        sorting = sortperm(vcat(constant_ics, tunable_ic))
        shape = size(u0)
        constants = constant_ics
        InitialConditionRemaker(sorting, constants)
    end
    # Setup the parameters
    p_vec, repack, _ = SciMLStructures.canonicalize(
        SciMLStructures.Tunable(), layer.problem.p
    )

    # We filter controls which do not act on the dynamics
    active_controls = control_indices .<= lastindex(p_vec)
    control_indices = control_indices[active_controls]
    controls = controls[active_controls]

    parameter_matrix = zeros(
        Bool, size(p_vec, 1), size(p_vec, 1) - size(control_indices, 1)
    )
    control_matrix = zeros(Bool, size(p_vec, 1), size(control_indices, 1))
    param_id = 0
    control_id = 0
    for i in eachindex(p_vec)
        if i ∈ control_indices
            control_matrix[i, control_id += 1] = true
        else
            parameter_matrix[i, param_id += 1] = true
        end
    end
    parameter_vector = let repack = repack, A = parameter_matrix, B = control_matrix
        function (params, controls)
            return repack(A * params .+ B * controls)
        end
    end

    # Next we setup the tspans and the indices
    if !isempty(controls)
        grid = build_index_grid(controls...; tspan, subdivide = 100)
        tspans = collect_tspans(controls...; tspan, subdivide = 100)
    else
        grid = Int64[i for i in control_indices]
        tspans = (problem.tspan,)
    end
    shooting_indices = zeros(Bool, size(u0, 1) + length(controls))
    if shooting_layer
        shooting_indices[eachindex(u0)] .= true
        for (i, c) in enumerate(controls)
            shooting_indices[lastindex(u0) + i] = !find_shooting_indices(first(tspans), c)
        end
    end
    shooting_indices = findall(shooting_indices)
    symcache = retrieve_symbol_cache(problem, control_indices, controls)
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

function (layer::SingleShootingLayer)(::Any, ps, st)
    (; initial_condition) = st
    u0 = initial_condition(ps.u0, layer.problem.u0)
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
    return Trajectory(sys, u, p, t, empty(u), Int64[])
end

sequential_solve(args...) = _sequential_solve(args...)

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
            push!(ex, :($(u0s[i + 1]) = last($(solutions[i]).u)[eachindex(u0)]))
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
                $(solutions[i]) = solve(
                    problem,
                    alg;
                    u0 = $(u0s[i]),
                    dense = false,
                    save_start = $(i == 1),
                    save_end = true,
                    tspan = tspans[$(i)],
                    p = param($(psym[i])),
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
    get_block_structure(layer)

Compute the block structure of the hessian of the Lagrangian of an optimal control problem.
As this is a `SingleShootingLayer`, this hessian is dense.
See also [``MultipleShootingLayer``](@ref).
"""
function get_block_structure(
        layer::SingleShootingLayer, tspan = layer.problem.tspan, kwargs...
    )
    return vcat(0, LuxCore.parameterlength(layer; tspan, kwargs...))
end
