struct SingleShootingLayer{P,A,C} <: LuxCore.AbstractLuxLayer
    problem::P
    algorithm::A
    tunable_ic::Vector{Int64}
    control_indices::Vector{Int64}
    controls::C
end

function LuxCore.initialparameters(rng::Random.AbstractRNG, layer::SingleShootingLayer)
    p_vec, _... = SciMLStructures.canonicalize(SciMLStructures.Tunable(), layer.problem.p)
    (;
        u0=copy(layer.problem.u0[layer.tunable_ic]),
        p=getindex(p_vec, [i for i in eachindex(p_vec) if i ∉ layer.control_indices]),
        controls=collect_local_controls(rng, layer.controls...)
    )
end

function LuxCore.initialstates(rng::Random.AbstractRNG, layer::SingleShootingLayer)
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
        function (u::AbstractArray{T}) where T
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
    grid = build_index_grid(controls...; tspan=problem.tspan, subdivide = 100)
    tspans = collect_tspans(controls...; tspan=problem.tspan, subdivide = 100)
    (;
        initial_condition,
        index_grid=grid,
        tspans,
        parameter_vector,
    )
end

function (layer::SingleShootingLayer)(::Any, ps, st)
    (; problem, algorithm,) = layer
    (; u0, p, controls) = ps
    (; index_grid, tspans, parameter_vector, initial_condition) = st
    u0_ = initial_condition(u0)
    params = Base.Fix1(parameter_vector, p)
    # Subdivide the grid and tspans 
    #N = length(tspans)
    #lower = 1:200:N
    #griddata = map(zip(lower, vcat(lower[2:end], N))) do (start, stop)
    #    index_grid[:, start:stop], tuple(collect(tspans)[start:stop]...)
    #end |> Tuple
    #solutions = batched_sequential_solve(problem, algorithm, u0_, params, controls, first.(griddata), last.(griddata))
    #return solutions
     solutions = sequential_solve(problem, algorithm, u0_, params, controls, index_grid, tspans)
    #    solutions = EnsembleSolution(solutions, 0.0, true, nothing)
    return solutions, st
end

@inline tuplejoin(x) = x

@inline tuplejoin(x, y) = (x..., y...)

@inline tuplejoin(x, y, z...) = (x..., tuplejoin(y, z...)...)


sequential_solve(args...) = _sequential_solve(args...)

@generated function _sequential_solve(problem, alg, u0, param, ps, indexgrids::NTuple{N}, tspans::NTuple{N, Tuple}) where N 
    solutions = [gensym() for _ in 1:N]
    u0s = [gensym() for _ in 1:N]
    ex = Expr[] 
    push!(ex, 
        :($(u0s[1]) = u0)
    )
    
    for i in 1:N 
        push!(ex, 
            :($(solutions[i]) = _sequential_solve(problem, alg, $(u0s[i]), param, ps, indexgrids[$(i)], tspans[$(i)]))
        )
        if i < N 
            push!(ex, 
                :($(u0s[i+1]) = last($(solutions[i]))[end])
            )
        end
    end
    push!(ex, :(return tuplejoin($(solutions...))))
    return Expr(:block, ex...)
end

@generated function _sequential_solve(problem, alg, u0, param, ps, index_grid::AbstractArray, tspans::NTuple{N, Tuple{<:Real, <:Real}}) where N
    solutions = [gensym() for _ in 1:N]
    u0s = [gensym() for _ in 1:N]
    ex = Expr[]
    push!(ex,
        :($(u0s[1]) = u0)
    )
    for i in 1:N
        push!(ex,
            :($(solutions[i]) = solve(problem, alg; u0=$(u0s[i]), dense=false, save_start=$(i == 1), save_end = true, tspan=tspans[$(i)], p=param(ps[index_grid[:, $(i)]])))
        )
        if i < N
            push!(ex,
                :($(u0s[i+1]) = $(solutions[i])[end])
            )
        end
    end
    push!(ex,
        :(return ($(solutions...),)) # Was kommt hier raus 
    )
    return Expr(:block, ex...)
end

