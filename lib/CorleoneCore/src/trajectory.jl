"""
$(TYPEDEF)

A container which collects the necessary information of a `DESolution`.

# Fields
$(FIELDS)
"""
struct Trajectory{U,P,T,S,R,MI,M}
    "The states of the trajectory"
    states::U
    "The parameters of the trajectory"
    parameters::P
    "The time points of the trajectory"
    time::T
    "The shooting states of the trajectory"
    shooting_variables::S
    "The returncodes"
    retcodes::R
    "Mayer indices"
    mayer_indices::MI
    "Meyer variables"
    mayer_variables::M
end

Base.eltype(x::Trajectory) = eltype(x.states)

function Trajectory(sol::DESolution; mayer::Union{AbstractVector,Nothing}=nothing)
    states = Array(sol)
    parameters = sol.prob.p
    time = sol.t
    _mayer = isnothing(mayer) ? nothing : states[mayer,end]
    Trajectory(states, parameters, time, nothing, sol.retcode == SciMLBase.ReturnCode.Success, mayer, _mayer)
end

function Base.merge(trajectories::Trajectory{U, P, T}...) where {U, P, T}
    N = length(trajectories)
    states = reduce(hcat, map(i -> i == N ? trajectories[i].states : trajectories[i].states[:, 1:end-1], eachindex(trajectories)))
    time = reduce(vcat, map(i -> i == N ? trajectories[i].time : trajectories[i].time[1:end-1], eachindex(trajectories)))
    retcodes = all(x -> x.retcodes, trajectories)
    mayers = [x.mayer_variables for x in trajectories]
    mayer_indices = first(trajectories).mayer_indices
    shooting_variables::Vector{NTuple{2, Vector{eltype(U)}}} = [(trajectories[i].states[.!mayer_indices, end], trajectories[i+1].states[.!mayer_indices, 1]) for i in  Base.OneTo(N - 1)]
    Trajectory{U, P, T, typeof(shooting_variables), typeof(retcodes), typeof(mayer_indices),
            typeof(mayers)}(states, trajectories[1].parameters, time, shooting_variables, retcodes, mayer_indices, mayers)
end
