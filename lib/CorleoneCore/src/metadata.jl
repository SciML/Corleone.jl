"""
$(TYPEDEF)

A struct to annotate tstops.
"""
struct Tstop end
Symbolics.option_to_metadata_type(::Val{:tstop}) = Tstop

is_tstop(x::Union{Num,Symbolics.Arr,SymbolicUtils.Symbolic}) = hasmetadata(x, Tstop)

function get_tstoppoints(sys, ::Type{T} = Float64) where T
    tstops_params = filter(is_tstop, parameters(sys))
    tstops_points = reduce(vcat, Symbolics.getdefaultval.(tstops_params), init = T[])
    unique!(sort!(tstops_points))
end

"""
$(TYPEDEF)

A struct to annotate local control.
"""
struct LocalControl end
Symbolics.option_to_metadata_type(::Val{:localcontrol}) = LocalControl

is_localcontrol(x::Union{Num,Symbolics.Arr,SymbolicUtils.Symbolic}) = hasmetadata(x, LocalControl)

"""
$(TYPEDEF)

A struct for storing the active shooting timepoints of a variable.
"""
struct ShootingTimepoint end
Symbolics.option_to_metadata_type(::Val{:shooting}) = ShootingTimepoint

is_shootingpoint(x::Union{Num,Symbolics.Arr,SymbolicUtils.Symbolic}) = hasmetadata(x, ShootingTimepoint)

function get_shootingpoints(sys, ::Type{T} = Float64) where T
    shooting_params = filter(is_shootingpoint, parameters(sys))
    shooting_points = reduce(vcat, Symbolics.getdefaultval.(shooting_params), init = T[])
    unique!(sort!(shooting_points))
end

"""
$(TYPEDEF)

A struct to annotate added variables for storing costs.
"""
struct CostVariable end
Symbolics.option_to_metadata_type(::Val{:costvariable}) = CostVariable
is_costvariable(x::Union{Num,Symbolics.Arr,SymbolicUtils.Symbolic}) = hasmetadata(x, CostVariable)
