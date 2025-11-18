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
struct SingleShootingLayer{P,A,C,B,PB} <: LuxCore.AbstractLuxLayer
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
  "Bounds on the tunable parameters of the problem"
  bounds_p::PB
end

function Base.show(io::IO, layer::SingleShootingLayer)
    type_color, no_color = SciMLBase.get_colorizers(io)

    print(io,
        type_color, "SingleShootingLayer",
        no_color, " with $(length(layer.controls)) controls.\nUnderlying problem: " )

    Base.show(io, "text/plain", layer.problem)
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
function SingleShootingLayer(prob, alg, control_indices=Int64[], controls=nothing; tunable_ic=Int64[], bounds_ic=nothing, bounds_p=nothing, kwargs...)
  _prob = init_problem(remake(prob; kwargs...), alg)
  u0 = prob.u0
  p_vec, _... = SciMLStructures.canonicalize(SciMLStructures.Tunable(), prob.p)
	p_vec = [p_vec[i] for i in eachindex(p_vec) if i ∉ control_indices]
  ic_bounds = isnothing(bounds_ic) ? (to_val(u0[tunable_ic], -Inf), to_val(u0[tunable_ic], Inf)) : bounds_ic
  p_bounds = isnothing(bounds_p) ? (to_val(p_vec, -Inf), to_val(p_vec, Inf)) : bounds_p
  return SingleShootingLayer(_prob, alg, control_indices, controls, tunable_ic, ic_bounds, p_bounds)
end

get_problem(layer::SingleShootingLayer) = layer.problem
get_controls(layer::SingleShootingLayer) = (layer.controls, layer.control_indices)
get_tspan(layer::SingleShootingLayer) = layer.problem.tspan
get_tunable(layer::SingleShootingLayer) = layer.tunable_ic
get_params(layer::SingleShootingLayer) = setdiff(eachindex(layer.problem.p), layer.control_indices)
__get_tunable_p(layer::SingleShootingLayer) = first(SciMLStructures.canonicalize(SciMLStructures.Tunable(), layer.problem.p))


function get_bounds(layer::SingleShootingLayer; kwargs...)
  (; bounds_ic, bounds_p, controls) = layer
  control_lb, control_ub = collect_local_control_bounds(controls...; kwargs...) 
	(
		(; u0 = first(bounds_ic), p = first(bounds_p), controls = control_lb), 
		(; u0 = last(bounds_ic), p = last(bounds_p), controls = control_ub)
	)
end

function LuxCore.initialparameters(rng::Random.AbstractRNG, layer::SingleShootingLayer; tspan=layer.problem.tspan, shooting_layer=false, kwargs...)
  p_vec, _... = SciMLStructures.canonicalize(SciMLStructures.Tunable(), layer.problem.p)
  (;
    u0=shooting_layer ? copy(layer.problem.u0) : copy(layer.problem.u0[layer.tunable_ic]),
    p=getindex(p_vec, [i for i in eachindex(p_vec) if i ∉ layer.control_indices]),
    controls=isnothing(layer.controls) ? eltype(layer.problem.u0)[] : collect_local_controls(rng, layer.controls...; tspan, kwargs...)
  )
end

function LuxCore.parameterlength(layer::SingleShootingLayer; tspan=layer.problem.tspan, shooting_layer=false, kwargs...)
  p_vec, _... = SciMLStructures.canonicalize(SciMLStructures.Tunable(), layer.problem.p)
  N = shooting_layer ? prod(size(layer.problem.u0)) : size(layer.tunable_ic, 1)
  N += sum([i ∉ layer.control_indices for i in eachindex(p_vec)])
  N += sum(Base.Fix2(control_length, tspan), layer.controls)
  return N
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


struct InitialConditionRemaker <: Function
  sorting::Vector{Int64}
  constants::Vector{Int64}
end

function (ic::InitialConditionRemaker)(u::AbstractVector{T}, u0::AbstractArray) where {T<:Number}
  (; constants, sorting) = ic
  isempty(u) && return T.(u0)
  reshape(vcat(vec(u0[constants]), u)[sorting], size(u0))
end

function (ic::InitialConditionRemaker)(::Any, u0::AbstractArray)
  u0
end

function LuxCore.initialstates(rng::Random.AbstractRNG, layer::SingleShootingLayer;
  tspan=layer.problem.tspan, shooting_layer=false, kwargs...)
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
  shooting_indices = zeros(Bool, size(u0, 1) + length(controls))
  if shooting_layer
    shooting_indices[eachindex(u0)] .= true
    for (i, c) in enumerate(controls)
      shooting_indices[lastindex(u0)+i] = !find_shooting_indices(first(tspans), c)
    end
  end
  shooting_indices = findall(shooting_indices)
  symcache = retrieve_symbol_cache(problem, control_indices)
  (;
    initial_condition,
    index_grid=grid,
    tspans,
    parameter_vector,
    symcache,
    shooting_indices
  )
end

function (layer::SingleShootingLayer)(::Any, ps, st)
  (; initial_condition) = st
  u0 = initial_condition(ps.u0, layer.problem.u0)
  layer(u0, ps, st)
end

function (layer::SingleShootingLayer)(u0::AbstractArray, ps, st)
  (; problem, algorithm,) = layer
  (; p, controls) = ps
  (; index_grid, tspans, parameter_vector, symcache) = st
  params = Base.Fix1(parameter_vector, p)
  # Returns the states as DiffEqArray
  solutions = sequential_solve(problem, algorithm, u0, params, controls, index_grid, tspans, symcache)
  return solutions, st
end

function build_optimal_control_solution(u, t, p, sys)
  Trajectory(sys, u, p, t, empty(u), Int64[])
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


function _parallel_solve(::Any, layer::SingleShootingLayer, u0, ps, st)
  @warn "Falling back to using `EnsembleSerial`" maxlog = 1
  _parallel_solve(EnsembleSerial(), layer, u0, ps, st)
end

__getidx(x, id) = x[id]
__getidx(x::NamedTuple, id) = getproperty(x, id)

function _parallel_solve(alg::SciMLBase.EnsembleAlgorithm, layer::SingleShootingLayer, u0, ps, st::NamedTuple{fields}) where {fields}
  args = ntuple(i -> (u0, __getidx(ps, fields[i]), __getidx(st, fields[i])), length(st)) |> collect
  mythreadmap(alg, Base.Splat(layer), args)
end

"""
    get_block_structure(layer)

Compute the block structure of the hessian of the Lagrangian of an optimal control problem.
As this is a `SingleShootingLayer`, this hessian is dense.
See also [``MultipleShootingLayer``](@ref).
"""
function get_block_structure(layer::SingleShootingLayer, tspan=layer.problem.tspan, kwargs...)
  vcat(0, LuxCore.parameterlength(layer; tspan))
end
