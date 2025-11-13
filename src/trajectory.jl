using SymbolicIndexingInterface

"""
$(TYPEDEF)
The solution to an optimal control problem. 
# Fields
$(FIELDS)
# Note 
If present, `shooting_points` contains a list of `Tuple`s `(timeseries_index, last_shooting_point)`.  
"""
struct Trajectory{S, U, P, T, C}
		"The symbolic system used for SymbolicIndexingInterface"
    sys::S
		"The state trajectory"
    u::U
		"The parameter values"
    p::P
		"The timepoints"
    t::T
		"The shooting points"
		shooting_points::C
end

SymbolicIndexingInterface.is_timeseries(::Type{<:Trajectory}) = Timeseries()
function SymbolicIndexingInterface.is_timeseries(::Type{<:Trajectory{
        S, U, P, Nothing}}) where {S, U, P}
    NotTimeseries()
end
SymbolicIndexingInterface.symbolic_container(fp::Trajectory) = fp.sys
SymbolicIndexingInterface.state_values(fp::Trajectory) = fp.u
SymbolicIndexingInterface.parameter_values(fp::Trajectory) = fp.p
SymbolicIndexingInterface.current_time(fp::Trajectory) = fp.t

utype(traj::Trajectory) = eltype(first(traj.u))
ttype(traj::Trajectory) = eltype(traj.t)

is_shooting_solution(::Trajectory) = false  
is_shooting_solution(::Trajectory{<:Any,<:Any,<:Any,<:Any,Nothing}) = false  

"""
$(FUNCTIONNAME)
Computes the shooting constraints over a given [Trajectory](@ref) if present. 
"""
shooting_constraints(x::Trajectory{<:Any,<:Any,<:Any,<:Any,Nothing}) = utype(x)[]
function shooting_constraints(traj::Trajectory)
	(; u, shooting_points) = traj
	reduce(vcat, map(shooting_points) do (idx, u_last) 
		vec(u_last .- u[idx])
	end)
end

"""
$(FUNCTIONNAME)
Computes the shooting constraints over a given [Trajectory](@ref) if present in place. 
"""
shooting_constraints!(res::AbstractArray, ::Trajectory{<:Any,<:Any,<:Any,<:Any,Nothing}) = nothing
function shooting_constraints!(res::AbstractVector, traj::Trajectory)
	(; u, shooting_points) = traj
	skip = prod(size(vec(u[1])))
	foreach(enumerate(shooting_points)) do (i, (idx, u_last)) 
		res[(i-1)*skip+1:i*skip] .= vec(u_last .- u[idx])
	end
end
