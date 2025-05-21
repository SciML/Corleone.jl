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

default_special_idx(sol::DESolution) = begin
    u0size = size(sol.prob.u0) 
    (; 
        shooting = ones(Bool, u0size),
        pseudo_mayer = zeros(Bool, u0size),
        controls = zeros(Bool, u0size)
    )
end

function Trajectory(sol::DESolution; special_variables::NamedTuple=default_special_idx(sol))
    states = Array(sol)
    parameters = sol.prob.p
    time = sol.t
    _mayer = states[special_variables.pseudo_mayer,end]
    Trajectory(states, parameters, time, nothing, sol.retcode == SciMLBase.ReturnCode.Success, special_variables, _mayer)
end

function Base.merge(trajectories::Trajectory...)
    N = length(trajectories)
    U = Base.promote_typeof(map(Base.Fix2(getfield, :states), trajectories)...)
    P = typeof(first(trajectories).parameters)
    T = Base.promote_typeof(map(Base.Fix2(getfield, :time), trajectories)...)
    states = reduce(hcat, map(i -> i == N ? trajectories[i].states : trajectories[i].states[:, 1:end-1], eachindex(trajectories)))
    time = reduce(vcat, map(i -> i == N ? trajectories[i].time : trajectories[i].time[1:end-1], eachindex(trajectories)))
    retcodes = all(x -> x.retcodes, trajectories)
    mayers = reduce(hcat, [x.mayer_variables for x in trajectories])
    special_variables = first(trajectories).special_variables
    shooting = special_variables.shooting 
    shooting_variables::Vector{NTuple{2, Vector{eltype(U)}}} = [(trajectories[i].states[shooting, end], trajectories[i+1].states[shooting, 1]) for i in  Base.OneTo(N - 1)]
    Trajectory{U, P, T, typeof(shooting_variables), typeof(retcodes), typeof(special_variables),
            typeof(mayers)}(states, trajectories[1].parameters, time, shooting_variables, retcodes, special_variables, mayers)
end
