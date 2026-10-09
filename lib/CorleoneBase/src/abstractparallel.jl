abstract type AbstractParallelProblem end

function get_problem(::AbstractParallelProblem) end
function get_problems end 
function prob_func(::AbstractParallelProblem, template, i) end

@inline function CommonSolve.solve(
    problem::T, alg::SciMLBase.AbstractSciMLAlgorithm, ensemblealg::SciMLBase.EnsembleAlgorithm;
    trajectories, 
    kwargs...
) where T <: AbstractParallelProblem

# Convert kwargs to a NamedTuple (type-stable)
    nt_kwargs = NamedTuple(kwargs)

    # Allowed keys as a static tuple of Symbols
    ensemble_keys = (:output_func, :reduction, :u_init, :safetycopy)

    # Keep only keys that actually exist in kwargs (Type-Stable via Base.structdiff)
    ensemblekwargs = Base.structdiff(nt_kwargs, Base.structdiff(nt_kwargs, NamedTuple{ensemble_keys}))
    
    # Or cleaner: extract ensemblekwargs directly using pure NamedTuple selection
    # ensemblekwargs = NamedTuple{intersect(keys(nt_kwargs), ensemble_keys)}(nt_kwargs)

    # solve_kwargs is simply the inverse diff
    solve_kwargs = Base.structdiff(nt_kwargs, ensemblekwargs)

    ensemble_problem = SciMLBase.EnsembleProblem(;
        prob = get_problem(problem),
        prob_func = Base.Fix1(prob_func, problem), 
        ensemblekwargs...
    )
    
    solve(ensemble_problem, alg, ensemblealg; trajectories, solve_kwargs...)
end