struct VariablePermutation
    "Forward permutation from the original order to the block order"
    fwd::Vector{Int64}
    "Reverse permutation from the block order to the original order"
    rev::Vector{Int64}
    "Array of variable indices (in permuted order) denoting the blocks
    in the Hessian of the Lagrangian of the discretized optimal control problem.
    Constructed for direct use with blockSQP."
    blocks::Vector{Int64}
end

transform(perm::VariablePermutation, p::AbstractVector) = getindex(p, perm.fwd)
invtransform(perm::VariablePermutation, p::AbstractVector) = getindex(p, perm.rev)

struct OCPredictor{N,P,A,E,S,T,K}
    "The underlying problem"
    problem::P
    "The algorithm to solve the problem"
    alg::A
    "The ensemble algorithm"
    ensemblealg::E
    "The shooting transition"
    shooting_transition::S
    "The shooting intervals"
    shooting_intervals::T
    "Solver kwargs"
    solver_kwargs::K
    "Permutation of variables to obtain block structure"
    permutation::VariablePermutation
end

OCPredictor(sys, alg, ensemblealg=EnsembleSerial(); kwargs...) = OCPredictor{true}(sys, alg, ensemblealg; kwargs...)

OCPredictor(problem, alg, ensemblealg, transition, intervals, kwargs, permutation) = OCPredictor{length(intervals),typeof(problem),typeof(alg),typeof(ensemblealg),typeof(transition),typeof(intervals),typeof(kwargs)}(problem, alg, ensemblealg, transition, intervals, kwargs, permutation)

function (f::AbstractNodeInitialization)(predictor::OCPredictor; kwargs...)
    newprob = f(predictor.problem, predictor.alg; kwargs...)
    @set predictor.problem = newprob
end

function find_control_pairs(sys, var)
    var = Symbolics.unwrap(var)
    x = Symbol(iscall(var) ? operation(var) : var)
    ps = ModelingToolkit.getvar(sys, Symbol(x, :ᵢ), namespace=false)
    ts = ModelingToolkit.getvar(sys, Symbol(x, :ₜ), namespace=false)
    [xi for xi in zip(collect(ps), collect(ts))]
end

function find_shooting_pairs(sys, var)
    var = Symbolics.unwrap(var)
    x = Symbol(iscall(var) ? operation(var) : var)
    ps = ModelingToolkit.getvar(sys, Symbol(x, :ₛ), namespace=false)
    ts = ModelingToolkit.getvar(sys, Symbol(x, :ₛ, :ₜ), namespace=false)
    [xi for xi in zip(collect(ps), collect(ts))]
end

function compare_timedependent_variables(x, y)
    a, t1 = x
    b, t2 = y
    # This sorts by time
    Symbolics.getdefaultval(t1) < Symbolics.getdefaultval(t2) && return true
    # Shooting preceede controls
    if Symbolics.getdefaultval(t1) == Symbolics.getdefaultval(t2)
        is_localcontrol(a) && is_localcontrol(b) && return true
        is_shootingvariable(a) && is_shootingvariable(b) && return true
        is_shootingvariable(a) && is_localcontrol(b) && return true
        return false
    end
    return false
end

# TODO Adapt this to include lifted parameters
is_local_parameter(x) = is_localcontrol(x)

function _find_blocks(vars)
    idx = Int64[]
    idx_cache = Int64[]
    current = firstindex(vars)
    pushable = true
    parent = nothing
    while current <= lastindex(vars)
        xi = vars[current]
        if is_shootingvariable(xi) && pushable
            # First parent variable. Should stay consistent
            if isnothing(parent)
                parent = get_shootingparent(xi)
            end
            if isequal(get_shootingparent(xi), parent)
                push!(idx_cache, current)
            end
            pushable = false
        elseif is_localcontrol(xi)
            # This is a control
            pushable = true
            append!(idx, idx_cache)
            empty!(idx_cache)
        end
        current += 1
    end
    idx .-= 1
    push!(idx, lastindex(vars))
    return idx
end

_maybecollect(x) = x
_maybecollect(x::SymbolicUtils.BasicSymbolic{<:AbstractArray}) = collect(x)

function compute_permutation_of_variables(sys)
    ps = filter(!ModelingToolkit.isinitial, reduce(vcat, _maybecollect.(tunable_parameters(sys))))
    ctrls = filter(ModelingToolkit.isinput, vcat(unknowns(sys), observables(sys)))
    states = filter(is_statevar, unknowns(sys))
    ctrl_vars = reduce(vcat, map(Base.Fix1(find_control_pairs, sys), ctrls), init=[])
    shooting_vars = reduce(vcat, map(Base.Fix1(find_shooting_pairs, sys), states))
    vars_to_sort = vcat(ctrl_vars, shooting_vars)
    sort!(vars_to_sort, lt=compare_timedependent_variables)
    vars = first.(vars_to_sort)
    rest = findall(x -> !any(Base.Fix1(isequal, x), vars), ps)
    new_order = vcat(vars, ps[rest])
    block_structure = isempty(rest) || (!isempty(rest) && all(is_regularization, ps[rest]))
    blocks_hess = if !block_structure
        [0, length(ps)]
    else
        _find_blocks(new_order)
    end

    perm = Int64[findfirst(x -> isequal(x, y), ps) for y in new_order]
    rev_perm = sortperm(perm)

    VariablePermutation(perm, rev_perm, blocks_hess)
end


function OCPredictor{IIP}(sys, alg, tspan, ensemblealg=EnsembleSerial(), args...; saveat=[], kwargs...) where {IIP}
    @assert !isnothing(tspan) "No tspan provided!"
    shooting_init = build_shooting_initializer(sys)
    shooting_timepoints = get_shootingpoints(sys)
    tspan = extrema(vcat(shooting_timepoints, collect(tspan)))
    shooting_intervals = collect(zip(shooting_timepoints[1:end-1], prevfloat.(shooting_timepoints[2:end])))
    if isempty(shooting_intervals)
        push!(shooting_intervals, tspan)
    elseif last(tspan) != last(shooting_timepoints)
        push!(shooting_intervals, (shooting_timepoints[end], last(tspan)))
    end
    append!(saveat, last.(shooting_intervals))
    tstops = get_tstoppoints(sys)
    sort!(saveat)
    unique!(saveat)
    perm = compute_permutation_of_variables(sys)
    kwargs = merge(NamedTuple(kwargs), (; tstops=tstops, saveat=saveat))
    problem = ODEProblem{IIP,SciMLBase.FullSpecialize}(sys, [], tspan; jac=true, tgrad=true, build_initializeprob=false, check_compatibility=false, kwargs...)
    return OCPredictor{
        length(shooting_intervals),typeof(problem),
        typeof(alg),typeof(ensemblealg),
        typeof(shooting_init),
        typeof(shooting_intervals),
        typeof(kwargs)
    }(
        problem, alg, ensemblealg, shooting_init, shooting_intervals, kwargs, perm
    )
end

function get_p0(predictor::OCPredictor)
    (; problem, permutation) = predictor
    p, _, _ = SciMLStructures.canonicalize(SciMLStructures.Tunable(), problem.p)
    transform(permutation, p)
end

function get_bounds(predictor::OCPredictor)
    (; permutation, problem) = predictor
    ps = SciMLBase.getparamsyms(problem)
    filter!(!ModelingToolkit.isinitial, ps)
    filter!(ModelingToolkit.istunable, ps)
    bounds = map(ps) do pi
        bi = ModelingToolkit.getbounds(pi)
        eltype(bi) <: Real ? ([bi[1]], [bi[2]]) : bi
    end
    lbounds = reduce(vcat, vec.(first.(bounds)))
    ubounds = reduce(vcat, vec.(last.(bounds)))
    transform(permutation, lbounds), transform(permutation, ubounds)
end

function (predictor::OCPredictor{1})(p::AbstractVector{T}; kwargs...) where {T}
    (; problem, alg, shooting_intervals, shooting_transition,
        permutation, solver_kwargs) = predictor
    _p = invtransform(permutation, p)
    new_params = SciMLStructures.replace(SciMLStructures.Tunable(), problem.p, _p)
    tspan = only(shooting_intervals)
    u0 = shooting_transition(problem.u0, new_params, first(tspan))
    new_problem = remake(problem, p=new_params, u0=u0, tspan=tspan)
    solve(new_problem, alg; merge(solver_kwargs, kwargs)...), new_params
end

function (predictor::OCPredictor{N})(p::AbstractVector; kwargs...) where {N}
    (; problem, alg, ensemblealg, shooting_transition,
        shooting_intervals, solver_kwargs, permutation) = predictor
    _p = invtransform(permutation, p)
    new_params = SciMLStructures.replace(SciMLStructures.Tunable(), problem.p, _p)
    probfunc = let shooting_intervals = shooting_intervals, shooting_init = shooting_transition, p = new_params
        function (prob, i, repeat)
            current_tspan = shooting_intervals[i]
            newu0 = shooting_init(prob.u0, p, first(current_tspan))
            remake(prob; u0=newu0, p=p, tspan=current_tspan)
        end
    end
    problem = EnsembleProblem(problem, prob_func=probfunc)
    sols = solve(problem, alg, ensemblealg, trajectories=N; merge(solver_kwargs, kwargs)...), new_params
end

function find_idx(full::AbstractVector, subset::AbstractVector)
    idx = Int64[]
    @inbounds for i in eachindex(subset)
        id = searchsortedlast(full, subset[i])
        (id in eachindex(full)) && (id ∉ idx)  && push!(idx, id)
    end
    return idx
end

# TODO This can go into an extension
using ChainRulesCore
ChainRulesCore.@non_differentiable find_idx(::AbstractVector, ::AbstractVector)

function predict(predictor::OCPredictor, p)
    sol, ps = predictor(p)
    # Reduce all solutions here
    collect_solution(sol), ps
end
# We filter the idx here
function collect_solution(sol::DESolution)
    idx = find_idx(sol.t, sol.prob.kwargs[:saveat])
    Array(sol)[:, idx]
end

function collect_solution(sol::EnsembleSolution)
    x = map(eachindex(sol)) do i
        collect_solution(sol.u[i])
    end
    reduce(hcat, i == 1 ? x[i] : x[i][:, 2:end] for i in eachindex(x))
end
