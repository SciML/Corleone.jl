using CommonSolve, SciMLBase
using OrdinaryDiffEqTsit5

using SciMLStructures

abstract type AbstractShootingProblem end

struct ShootingProblem{P<:SciMLBase.DEProblem,U,C} <: AbstractShootingProblem
    problem::P
    u0::U
    controls::C
end

get_defaults(prob::ShootingProblem) = collect_parameters(prob.problem, prob.u0, prob.controls).p

struct ShootingPredictor{P,Q,U,S,A,K}
    problem::P
    p0::Q
    u0map::U
    sequence::S
    algorithm::A
    kwargs::K
end


function CommonSolve.init(prob::ShootingProblem, alg::SciMLBase.AbstractDEAlgorithm; kwargs...)
    (; problem, u0, controls) = prob
    conf = collect_parameters(problem, u0, controls)
    ShootingPredictor(problem, conf.p, conf.u0, conf.controlsequence, alg, kwargs)
end

function CommonSolve.solve!(prob::ShootingPredictor)
    (; problem, p0, u0map, sequence, algorithm, kwargs) = prob
    p = get(kwargs, :p, p0)
    u0 = get(kwargs, :u0, u0map)
    ensemble_sequential_solve(problem, algorithm, p, u0, sequence)
end

replace_portion(x, y, ::Any) = x

function replace_portion(x::AbstractArray{X}, y::AbstractArray{Y}, indices::Vector{Pair{Int,Int}}) where {X, Y}
    T = promote_type(X, Y)
    isempty(indices) && return T.(x)
    a, b = first.(indices), last.(indices)
    xreplace = [i ∈ b for i in eachindex(x)]
    xkeep = .! xreplace 
    ys = y[a]
    xkeep .* x .+ xreplace .* ys
    #finder = Base.Fix1(searchsortedfirst, b)
    #x_replaced = T.(map(eachindex(x)) do i
    #    idx = finder(i)
    #    idx > lastindex(a) && return x[i]
    #    y_idx = a[idx]
    #    y[y_idx]
    #end)
end

#==
function __replace_portion(x::AbstractArray, y::AbstractArray, indices::Vector{Pair{Int,Int}})
    replacement_map = Dict{Int, eltype(y)}()
    sizehint!(replacement_map, length(indices))
    for (yi, xi) in indices
        replacement_map[xi] = y[yi]
    end
    return [get(replacement_map, i, x[i]) for i in eachindex(x)]
end
==#


# No initial condition
function sequential_solve(problem::SciMLBase.DEProblem, alg::SciMLBase.DEAlgorithm, q::AbstractArray, ::Nothing, sequences::Tuple)
    sequential_solve(problem, alg, q, problem.u0, sequences)
end

# Tunable initial condition
function sequential_solve(problem::SciMLBase.DEProblem, alg::SciMLBase.DEAlgorithm, q::AbstractArray, u0idx::Vector{<:Pair}, sequences::Tuple)
    u0 = replace_portion(problem.u0, q, u0idx)
    sequential_solve(problem, alg, q, u0, sequences)
end

# TODO This does not handle the SciMLStructures interface. However, we just assume that we have a flat structure. (Or a wrapped type)
# Controls
function sequential_solve(problem::SciMLBase.DEProblem, alg::SciMLBase.DEAlgorithm, q::AbstractArray, u0, sequences::Tuple)
    (idx, tspan) = Base.first(sequences)
    new_problem = remake(problem, u0=u0, p=replace_portion(problem.p, q, idx), tspan=tspan)
    solution = solve(new_problem, alg)
    return (solution, sequential_solve(new_problem, alg, q, solution.u[end], Base.tail(sequences))...)
end

# Last control
function sequential_solve(problem::SciMLBase.DEProblem, alg::SciMLBase.DEAlgorithm, q::AbstractArray, u0, ::Tuple{})
    return ()
end

# TODO Maybe use me !?
function ensemble_sequential_solve(problem::SciMLBase.DEProblem, alg::SciMLBase.DEAlgorithm, q::AbstractArray, u0, sequences::Tuple)
    #t0 = time()
    solutions = reduce(vcat, sequential_solve(problem, alg, q, u0, sequences))
    t = 0.0 #time() - t0
    converged = all(SciMLBase.successful_retcode, solutions)
    EnsembleSolution(solutions, (length(sequences),), t, converged, nothing)
end

function check_config(problem, (idx, config)::Pair)
    @assert idx ∈ eachindex(problem.p)
    @assert size(config.timepoints) == size(config.defaults)
    return true
end

function collect_parameters(problem, config, poffset::Int=0)
    stuff = reduce(vcat, map(config) do (idx, conf)
        [(ti, idx, vi) for (ti, vi) in zip(conf.timepoints, conf.defaults)]
    end)
    sort!(stuff, by=Base.Fix2(getindex, 1:2))
    q = last.(stuff)
    # Merge common elements 
    timepoints = first.(stuff)
    append!(timepoints, collect(problem.tspan))
    unique!(sort!(timepoints))
    indexsets = map(enumerate(timepoints)) do (i, ti)
        idxs = findall(x -> first(x) == ti, stuff)
        Pair{Int,Int}[id + poffset => stuff[id][2] for id in idxs], (ti, i < lastindex(timepoints) ? timepoints[i+1] : last(problem.tspan))
    end
    q, tuple(indexsets...)
end

function collect_parameters(u0map)
    u0map = sort(u0map, by=first)
    idxs = first.(u0map)
    q = last.(u0map)
    q, [j => i for (i, j) in enumerate(idxs)]
end


u0 = [1 => 1.0,]


function collect_parameters(prob, u0map, configuration)
    qu0, idx0 = collect_parameters(u0map)
    qu, idxs = collect_parameters(prob, configuration, length(qu0))
    (; p=vcat(qu0, qu), u0=idx0, controlsequence=idxs)
end

# Test
function lotka(u, p, t)
    [u[1] - u[1] * u[2] - p[1] * u[1], u[1] * u[2] - u[2] - p[2] * u[2]]
end

u0 = ones(2)
p0 = zeros(2)
tspan = (0.0, 10.0)

prob = ODEProblem{false, SciMLBase.FullSpecialize}(lotka, u0, tspan, p0)

using OrdinaryDiffEqTsit5

sol = solve(prob, Tsit5())

shoot = ShootingProblem(
    prob,
    # Variable initial condition 1 
    [1 => 3.0],
    # Control specs for 1 and 2
    (
        1 => (; defaults=zeros(5), timepoints=LinRange(3.0, 10.0, 5)),
        2 => (; defaults=zeros(10), timepoints=LinRange(3.0, 5.0, 10)),
    )
)

p0 = get_defaults(shoot)

# First Call 
@time sol_1 = solve(shoot, Tsit5())
# Second Call 
@time sol_1 = solve(shoot, Tsit5())
plot(sol_1)

p0[2:end] .= Float64.(rand(Bool, length(p0) - 1))
sol = solve(shoot, Tsit5(), p=p0)
@info sol.elapsedTime
plot(sol)

using SciMLSensitivity #, Zygote

objective = let shoot = shoot 
    (p) -> begin 
        sol = solve(prob, Tsit5(), p = p)
        sum(abs2, 1 .- last(sol.u))
    end 
end

objective(p0)

using ForwardDiff

ForwardDiff.gradient(objective, p0[1:2])

@btime ForwardDiff.gradient($objective, $p0)

using Zygote
Zygote.gradient(objective, p0[1:2])

Zygote.gradient(sum ∘ replace_portion, ones(2), randn(3), [2 =>2])

