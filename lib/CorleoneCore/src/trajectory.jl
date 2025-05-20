"""
$(TYPEDEF)

A container which collects the necessary information of a `DESolution`.

# Fields
$(FIELDS)
"""
struct Trajectory{U,P,T,S,R,SI,M}
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
    "Indices of variables for pseudo-Mayer costs or controls."
    special_variables::SI
    "Evaluated pseudo-mayer costs on each shooting interval following
        the transformation of the Lagrange term."
    mayer_variables::M
end

Base.eltype(x::Trajectory) = eltype(x.states)

function Trajectory(sol::DESolution; special_variables::Union{NamedTuple,Nothing}=nothing)
    states = Array(sol)
    parameters = sol.prob.p
    time = sol.t
    _mayer = isnothing(special_variables) ? nothing : states[special_variables.pseudo_mayer,end]
    Trajectory(states, parameters, time, nothing, sol.retcode == SciMLBase.ReturnCode.Success, special_variables, _mayer)
end

function Base.merge(trajectories::Trajectory{U, P, T}...) where {U, P, T}
    N = length(trajectories)
    states = reduce(hcat, map(i -> i == N ? trajectories[i].states : trajectories[i].states[:, 1:end-1], eachindex(trajectories)))
    time = reduce(vcat, map(i -> i == N ? trajectories[i].time : trajectories[i].time[1:end-1], eachindex(trajectories)))
    retcodes = all(x -> x.retcodes, trajectories)
    mayers = [x.mayer_variables for x in trajectories]
    special_variables = first(trajectories).special_variables
    shooting = .!(special_variables.pseudo_mayer .|| special_variables.control)
    shooting_variables::Vector{NTuple{2, Vector{eltype(U)}}} = [(trajectories[i].states[shooting, end], trajectories[i+1].states[shooting, 1]) for i in  Base.OneTo(N - 1)]
    Trajectory{U, P, T, typeof(shooting_variables), typeof(retcodes), typeof(special_variables),
            typeof(mayers)}(states, trajectories[1].parameters, time, shooting_variables, retcodes, special_variables, mayers)
end
