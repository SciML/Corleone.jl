struct VariablePermutation{P,B,H}
    "Forward permutation from the original order to the block order"
    fwd::P
    "Reverse permutation from the block order to the original order"
    rev::P
    "Variable bounds in original order"
    bounds_original::B
    "Variable bounds in permuted order"
    bounds_permuted::B
    "Array of variable indices (in permuted order) denoting the blocks
    in the Hessian of the Lagrangian of the discretized optimal control problem.
    Constructed for direct use with blockSQP."
    blocks::H
end

struct OCPredictor{N,P,A,E,I,S,T,K,M}
    "The underlying problem"
    problem::P
    "The algorithm to solve the problem"
    alg::A
    "The ensemble algorithm"
    ensemblealg::E
    "The initial conditions"
    initial_condition::I
    "The shooting transition"
    shooting_transition::S
    "The shooting intervals"
    shooting_intervals::T
    "Solver kwargs"
    solver_kwargs::K
    "Permutation of variables to obtain block structure"
    permutation::VariablePermutation
    "Indices of special variables that need to be treated differently in the evaluation:
        1) pseudo-Mayer variables on each shooting interval needed due to transformation
                of the Lagrange term, and
        2) control variables added via DirectControlCallbacks"
    special_variables::M
end


OCPredictor(sys, alg, ensemblealg=EnsembleSerial(); kwargs...) = OCPredictor{true}(sys, alg, ensemblealg; kwargs...)

function compute_permutation_of_variables(sys, shooting_intervals)
    ns = length(shooting_intervals) -1
    ps = parameters(sys)

    tunable_stuff = filter(istunable, ps)
    original_order = reduce(vcat, collect.(tunable_stuff))

    idx_states = map(x -> endswith(x, "ₛ"), string.(tunable_stuff))
    idx_controls = map(x -> endswith(x, "ᵢ"), string.(tunable_stuff))

    ctls = tunable_stuff[idx_controls]
    sts = tunable_stuff[idx_states]

    # get timepoints for switches of controls
    controls = string.(ctls)

    indices_strings_ctls = collect.(eachindex.(controls))
    var_contr_time = filter!(!isnothing, [length(indices_strings_ctls[i]) > 1 ? x[indices_strings_ctls[i][1:end-1]] * "ₜ" : nothing for (i, x) in enumerate(controls)])
    dfs_sys = defaults(sys)
    keys_dfs = collect(keys(dfs_sys))
    idx_ctrl_times = [findfirst(x -> x == y, string.(keys_dfs)) for y in var_contr_time]
    ctrl_times = [dfs_sys[keys_dfs[idx]] for idx in idx_ctrl_times]
    shooting_times = unique(last.(shooting_intervals))

    # This may cause trouble when the shooting grid or the control grids are initialized
    # via LinRange(), which may cause inaccurate representation of floating points
    # TODO: Make block identification more robust!s
    blocks_ctr = [[findfirst(tf -> x < tf, shooting_times) for x in ctrl_time] for ctrl_time in ctrl_times]
    first_blocks = [vcat([collect(x)[i] for x in sts]..., [collect(x)[bl.==i+1] for (x, bl) in zip(ctls, blocks_ctr)]...) for i = 1:ns]
    last_block = [collect(x)[ns+1] for x in sts]

    order = [first_blocks..., last_block]
    @info order ModelingToolkit.getbounds.(order)

    blocks_hess = vcat(0, reduce(vcat, cumsum(length(vcat(order[1:i]...))) for i = 1:length(order)))

    new_order = reduce(vcat, order)
    bounds_perm = getbounds.(new_order)
    bounds = getbounds.(original_order)

    perm = [findfirst(x -> isequal(x, y), original_order) for y in new_order]
    rev_perm = [findfirst(x -> isequal(x, y), new_order) for y in original_order]
    VariablePermutation{typeof(perm), typeof(bounds), typeof(blocks_hess)}(perm, rev_perm, bounds, bounds_perm, blocks_hess)
end


function OCPredictor{IIP}(sys, alg, ensemblealg=EnsembleSerial(); kwargs...) where {IIP}
    tspan = ModelingToolkit.get_tspan(sys)
    @assert !isnothing(tspan) "No tspan provided!"
    problem = ODEProblem{IIP}(sys, [], tspan, [], build_initializeprob=false, allow_cost=true)
    u0 = build_u0_initializer(sys)
    shooting_init = build_shooting_initializer(sys)
    shooting_timepoints = get_shootingpoints(sys)
    shooting_intervals = vcat((first(tspan), first(tspan)), collect(xi for xi in zip(shooting_timepoints[1:end-1], shooting_timepoints[2:end])))
    if isempty(shooting_intervals)
        push!(shooting_intervals, tspan)
    end
    perm = compute_permutation_of_variables(sys, shooting_intervals)
    mayer_indices = is_costvariable.(unknowns(sys))
    control_indices = isinput.(unknowns(sys))
    special_variables = (; shooting = .!(mayer_indices .|| control_indices), pseudo_mayer = mayer_indices, control=control_indices)
    return OCPredictor{
        length(shooting_intervals),typeof(problem),
        typeof(alg),typeof(ensemblealg),
        typeof(u0),typeof(shooting_init),
        typeof(shooting_intervals),
        typeof(kwargs), typeof(special_variables)
    }(
        problem, alg, ensemblealg, u0, shooting_init, shooting_intervals, kwargs, perm, special_variables
    )
end
function get_p0(predictor::OCPredictor; permute=false)
    (; problem, permutation) = predictor
    p, _, _ = SciMLStructures.canonicalize(SciMLStructures.Tunable(), problem.p)
    permute || return p
    p[permutation.fwd]
end

function get_bounds(predictor::OCPredictor; permute=false)
    (; permutation) = predictor
    bounds = !permute ? permutation.bounds_original : permutation.bounds_permuted
    Float64.(first.(bounds)), Float64.(last.(bounds))
end

function predict(predictor::OCPredictor{1}, p; permute=false, kwargs...)
    (; problem, alg, initial_condition, shooting_intervals, solver_kwargs,
            permutation, special_variables) = predictor
    _p = permute ? view(p, permutation.rev) : view(p,:)
    new_params = SciMLStructures.replace(SciMLStructures.Tunable(), problem.p, _p)
    tspan = only(shooting_intervals)
    u0 = initial_condition(problem.u0, new_params, first(tspan))
    new_problem = remake(problem, p=new_params, u0=u0, tspan=tspan)
    Trajectory(solve(new_problem, alg; solver_kwargs...); special_variables=special_variables)
end

function predict(predictor::OCPredictor{N}, p; permute=false, kwargs...) where {N}
    (; problem, alg, ensemblealg, initial_condition, shooting_transition,
        shooting_intervals, solver_kwargs, permutation, special_variables) = predictor
    _p = permute ? view(p, permutation.rev) : view(p,:)
    new_params = SciMLStructures.replace(SciMLStructures.Tunable(), problem.p, _p)
    probfunc = let shooting_intervals = shooting_intervals, u0 = initial_condition, shooting_init = shooting_transition, p = new_params
        function (prob, i, repeat)
            current_tspan = shooting_intervals[i]
            newu0 = i == 1 ? u0(prob.u0, p, first(current_tspan)) : shooting_init(prob.u0, p, first(current_tspan))
            remake(prob; u0=newu0, p=p, tspan=current_tspan)
        end
    end
    problem = EnsembleProblem(problem, prob_func=probfunc, output_func = (sol, i) -> (Trajectory(sol; special_variables=special_variables), false))
    sols = solve(problem, alg, ensemblealg, trajectories=N; solver_kwargs...)
    merge(sols.u...)
end

function (predictor::OCPredictor)(p; kwargs...)
    (; special_variables) = predictor
    sol = predict(predictor, p; kwargs...)
    collect_trajectory(sol, special_variables)
end

collect_trajectory(traj::Trajectory, x) = traj

function collect_trajectory(sol::DESolution, mayer)
    Trajectory(sol, mayer=mayer)
end

function collect_trajectory(sol::EnsembleSolution, mayer)
    trajs = map(sol.u) do subsol
        collect_trajectory(subsol, mayer)
    end
    merge(trajs...)
end
