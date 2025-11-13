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
struct SingleShootingLayer{P,A,C,B} <: LuxCore.AbstractLuxLayer
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
end

function init_problem(prob, alg)
    remake_problem(prob, SciMLBase.init(prob, alg))
end

function remake_problem(prob::ODEProblem, state)
    remake(prob, u0=state.u)
end

function remake_problem(prob::DAEProblem, state)
    remake(prob, u0=state.u, du0=state.du)
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
function SingleShootingLayer(prob, alg, control_indices=Int64[], controls=nothing; tunable_ic=Int64[], bounds_ic=nothing, kwargs...)
    _prob = init_problem(remake(prob; kwargs...), alg)
    return SingleShootingLayer(_prob, alg, control_indices, controls, tunable_ic, bounds_ic)
end

get_problem(layer::SingleShootingLayer) = layer.problem
get_controls(layer::SingleShootingLayer) = (layer.controls, layer.control_indices)
get_tspan(layer::SingleShootingLayer) = layer.problem.tspan
get_tunable(layer::SingleShootingLayer) = layer.tunable_ic
get_params(layer::SingleShootingLayer) = setdiff(eachindex(layer.problem.p), layer.control_indices)

_get_bounds(layer::SingleShootingLayer, lower::Bool=true) = begin
    p_vec, _... = SciMLStructures.canonicalize(SciMLStructures.Tunable(), layer.problem.p)
    (;
        u0=isnothing(layer.bounds_ic) ? eltype(layer.problem.u0)[] : (lower ? copy(layer.bounds_ic[1]) : copy(layer.bounds_ic[2])),
        p=getindex(p_vec, [i for i in eachindex(p_vec) if i ∉ layer.control_indices]),
        controls=isnothing(layer.controls) ? eltype(layer.problem.u0)[] : collect_local_control_bounds(lower, layer.controls...)
    )
end

get_bounds(layer::SingleShootingLayer) = begin
    lb = _get_bounds(layer, true)
    ub = _get_bounds(layer, false)
    return ComponentArray(lb), ComponentArray(ub)
end

function LuxCore.initialparameters(rng::Random.AbstractRNG, layer::SingleShootingLayer)
    p_vec, _... = SciMLStructures.canonicalize(SciMLStructures.Tunable(), layer.problem.p)
    (;
        u0=copy(layer.problem.u0[layer.tunable_ic]),
        p=getindex(p_vec, [i for i in eachindex(p_vec) if i ∉ layer.control_indices]),
        controls=isnothing(layer.controls) ? eltype(layer.problem.u0)[] : collect_local_controls(rng, layer.controls...)
    )
end

function retrieve_symbol_cache(problem::SciMLBase.DEProblem, control_indices)
    retrieve_symbol_cache(problem.f.sys, problem.u0, problem.p, control_indices)
end

function retrieve_symbol_cache(::Nothing, u0, p, control_indices)
    p0, _ = SciMLStructures.canonicalize(SciMLStructures.Tunable(), p)
    state_symbols = [Symbol(:x, Symbol(Char(0x2080 + i))) for i in eachindex(u0)]
    u_id = 0
    p_id = 0
    parameter_symbols = [i ∈ control_indices ? Symbol(:u, Symbol(Char(0x2080 + (u_id += 1)))) : Symbol(:p, Symbol(Char(0x2080 + (p_id += 1)))) for i in eachindex(p0)]
    tsym = [:t]
    _retrieve_symbol_cache(state_symbols, parameter_symbols, tsym, control_indices)
end

function retrieve_symbol_cache(cache::SymbolCache, u0, p, control_indices)
    psym = parameter_symbols(cache)
    vsym = variable_symbols(cache)
    sort!(psym, by=xi -> SymbolicIndexingInterface.parameter_index(cache, xi))
    sort!(vsym, by=xi -> SymbolicIndexingInterface.variable_index(cache, xi))
    _retrieve_symbol_cache(
        vsym,
        psym,
        independent_variable_symbols(cache),
        control_indices
    )
end

function _retrieve_symbol_cache(
    xs, ps, t, idx
)
    nonidx = filter(∉(idx), eachindex(ps))
    SymbolCache(vcat(xs, ps[idx]), ps[nonidx], t)
end

function LuxCore.initialstates(rng::Random.AbstractRNG, layer::SingleShootingLayer, tspan = layer.problem.tspan)
    (; tunable_ic, control_indices, problem, controls) = layer
    # We first derive the initial condition function
    constant_ic = [i ∉ tunable_ic for i in eachindex(problem.u0)]
    tunable_matrix = zeros(Bool, size(problem.u0, 1), size(tunable_ic, 1))
    id = 0
    for i in axes(tunable_matrix, 1)
        if i ∈ tunable_ic
            tunable_matrix[i, id+=1] = true
        end
    end
    initial_condition = let B = tunable_matrix, u0 = constant_ic .* copy(problem.u0)
        function (u::AbstractArray{T}) where {T}
            T.(u0) .+ B * u
        end
    end
    # Setup the parameters
    p_vec, repack, _ = SciMLStructures.canonicalize(SciMLStructures.Tunable(), layer.problem.p)
    parameter_matrix = zeros(Bool, size(p_vec, 1), size(p_vec, 1) - size(control_indices, 1))
    control_matrix = zeros(Bool, size(p_vec, 1), size(control_indices, 1))
    param_id = 0
    control_id = 0
    for i in eachindex(p_vec)
        if i ∈ control_indices
            control_matrix[i, control_id+=1] = true
        else
            parameter_matrix[i, param_id+=1] = true
        end
    end
    parameter_vector = let repack = repack, A = parameter_matrix, B = control_matrix
        function (params, controls)
            repack(A * params .+ B * controls)
        end
    end
    # Next we setup the tspans and the indices
    grid = build_index_grid(controls...; tspan, subdivide=100)
    tspans = collect_tspans(controls...; tspan, subdivide=100)
    symcache = retrieve_symbol_cache(problem, control_indices)
    (;
        initial_condition,
        index_grid=grid,
        tspans,
        parameter_vector,
        symcache
    )
end

function (layer::SingleShootingLayer)(::Nothing, ps, st)
    (; problem, algorithm,) = layer
    (; u0, p, controls) = ps
    (; index_grid, tspans, parameter_vector, initial_condition, symcache) = st
    u0_ = initial_condition(u0)
    params = Base.Fix1(parameter_vector, p)
    # Returns the states as DiffEqArray
    solutions = sequential_solve(problem, algorithm, u0_, params, controls, index_grid, tspans, symcache)
    return solutions, st
end

function (layer::SingleShootingLayer)(u0, ps, st)
    (; problem, algorithm,) = layer
    (; p, controls) = ps
    (; index_grid, tspans, parameter_vector, symcache) = st
    params = Base.Fix1(parameter_vector, p)
    # Returns the states as DiffEqArray
    solutions = sequential_solve(problem, algorithm, u0, params, controls, index_grid, tspans, symcache)
    return solutions, st
end

function build_optimal_control_solution(u, t, p, sys)
	Trajectory(sys, u, p, t, nothing)  
end

sequential_solve(args...) = _sequential_solve(args...)

@generated function _sequential_solve(problem, alg, u0, param, ps, indexgrids::NTuple{N}, tspans::NTuple{N,Tuple}, sys) where {N}
    solutions = [gensym() for _ in 1:N]
    u0s = [gensym() for _ in 1:N]
    ex = Expr[]
    u_ret_expr = :(vcat())
    t_ret_expr = :(vcat())
    push!(ex,
        :($(u0s[1]) = u0)
    )

    for i in 1:N
        push!(ex,
            :($(solutions[i]) = _sequential_solve(problem, alg, $(u0s[i]), param, ps, indexgrids[$(i)], tspans[$(i)], sys))
        )
#        push!(u_ret_expr.args, :($(solutions[i]).u))
#        push!(t_ret_expr.args, :($(solutions[i]).t))
        if i < N
        push!(u_ret_expr.args, :($(solutions[i]).u[1:end-1]))
        push!(t_ret_expr.args, :($(solutions[i]).t[1:end-1]))
            push!(ex,
                :($(u0s[i+1]) = last($(solutions[i]).u)[eachindex(u0)])
            )
        else

        push!(u_ret_expr.args, :($(solutions[i]).u))
        push!(t_ret_expr.args, :($(solutions[i]).t))
        end
    end
    push!(ex,
        :(return build_optimal_control_solution($(u_ret_expr), $(t_ret_expr), param.x, sys)) # Was kommt hier raus
    )
    return Expr(:block, ex...)
end


@generated function _sequential_solve(problem, alg, u0, param, ps, index_grid::AbstractArray, tspans::NTuple{N,Tuple{<:Real,<:Real}}, sys) where {N}
    solutions = [gensym() for _ in 1:N]
    u0s = [gensym() for _ in 1:N]
    ex = Expr[]
    u_ret_expr = :(vcat())
    t_ret_expr = :(vcat())
    psym = [gensym() for _ in 1:N]
    push!(ex,
        :($(u0s[1]) = u0)
    )
    for i in 1:N
        push!(ex, :($(psym[i]) = getindex(ps, index_grid[:, $(i)])))
        push!(ex,
            :($(solutions[i]) = solve(problem, alg; u0=$(u0s[i]), dense=false, save_start=$(i == 1), save_end=true, tspan=tspans[$(i)], p=param($(psym[i]))))
        )
        push!(u_ret_expr.args, :(Base.Fix2(vcat, $(psym[i])).($(solutions[i]).u)))
        push!(t_ret_expr.args, :($(solutions[i]).t))
        if i < N
            push!(ex,
                :($(u0s[i+1]) = $(solutions[i]).u[end])
            )
        end
    end
    push!(ex,
        :(return build_optimal_control_solution($(u_ret_expr), $(t_ret_expr), param.x, sys)) # Was kommt hier raus
    )
    return Expr(:block, ex...)
end


"""
$(TYPEDEF)

Prototypical single shooting problem to interface `CommonSolve` with.

# Fields
$(FIELDS)
"""
struct SingleShootingProblem{L,P,S}
    layer::L
    params::P
    state::S
end

function SingleShootingSolution(sols::NTuple)
    states = reduce(hcat, map(Array, sols))
    t = reduce(vcat, map(x -> x.t, sols))
    SingleShootingSolution{typeof(states),typeof(t)}(states, t)
end

struct DummySolve end

function CommonSolve.init(prob::SingleShootingProblem, ::DummySolve; kwargs...)
    prob
end

function CommonSolve.init(prob::SingleShootingProblem, ::Any; kwargs...)
    prob
end

function CommonSolve.solve!(prob::SingleShootingProblem)
    prob.layer(nothing, prob.params, prob.state)
end

function SciMLBase.remake(prob::SingleShootingProblem; ps=prob.params, st=prob.state)
    SingleShootingProblem(prob.layer, ps, st)
end

"""
    get_block_structure(layer)

Compute the block structure of the hessian of the Lagrangian of an optimal control problem.
As this is a `SingleShootingLayer`, this hessian is dense.
See also [``MultipleShootingLayer``](@ref).
"""
function get_block_structure(layer::SingleShootingLayer)
    vcat(0, LuxCore.parameterlength(layer))
end
