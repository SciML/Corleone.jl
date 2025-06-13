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
get_localcontrols(sys) = filter(is_localcontrol, parameters(sys))

"""
$(TYPEDEF)

A struct for storing the active shooting timepoints of a variable.
"""
struct ShootingTimepoint end
Symbolics.option_to_metadata_type(::Val{:shooting}) = ShootingTimepoint
is_shootingpoint(x::Union{Num,Symbolics.Arr,SymbolicUtils.Symbolic}) = hasmetadata(x, ShootingTimepoint)
get_shootingpoint_variables(sys) = filter(is_shootingpoint, parameters(sys))

function get_shootingpoints(sys, ::Type{T} = Float64) where T
    shooting_params = filter(is_shootingpoint, parameters(sys))
    shooting_points = reduce(vcat, Symbolics.getdefaultval.(shooting_params), init = T[])
    unique!(sort!(shooting_points))
end

"""
$(TYPEDEF)

A struct for storing the active shooting variables of a system and their respective parent.
"""
struct ShootingVariable end
Symbolics.option_to_metadata_type(::Val{:shooting_variable}) = ShootingVariable
is_shootingvariable(x::Union{Num,Symbolics.Arr,SymbolicUtils.Symbolic}) = hasmetadata(x, ShootingVariable)

function get_shootingvariables(sys, ::Type{T} = Float64) where T
    filter(is_shootingvariable, parameters(sys))
end

get_shootingparent(x::Num) = get_shootingparent(Symbolics.unwrap(x))
get_shootingparent(x) = Symbolics.getmetadata(x, ShootingVariable, x)

"""
$(TYPEDEF)

A struct to annotate added variables for storing costs.
"""
struct CostVariable end
Symbolics.option_to_metadata_type(::Val{:costvariable}) = CostVariable
is_costvariable(x::Union{Num,Symbolics.Arr,SymbolicUtils.Symbolic}) = hasmetadata(x, CostVariable)

is_statevar(x) = !is_costvariable(x) && !ModelingToolkit.isinput(x)

"""
A struct to annotate added variables for sensitivities.
"""
struct Sensitivities end
Symbolics.option_to_metadata_type(::Val{:sensitivities}) = Sensitivities
is_sensitivity(x::Union{Num,Symbolics.Arr,SymbolicUtils.Symbolic}) = hasmetadata(x, Sensitivities)

"""
A struct to annotate added variables for Fisher information matrices.
"""
struct FIM end
Symbolics.option_to_metadata_type(::Val{:fim}) = FIM
is_fim(x::Union{Num,Symbolics.Arr,SymbolicUtils.Symbolic}) = hasmetadata(x, FIM)

"""
A struct to annotate added variables for measurement variables for OED.
"""
struct Measurement end
Symbolics.option_to_metadata_type(::Val{:measurements}) = Measurement
is_measurement(x::Union{Num,Symbolics.Arr,SymbolicUtils.Symbolic}) = hasmetadata(x, Measurement)

"""
A struct to annotate parameters to consider for OED.
"""
struct UncertainParameter end
Symbolics.option_to_metadata_type(::Val{:uncertain}) = UncertainParameter
is_uncertain(x::Union{Num,Symbolics.Arr,SymbolicUtils.Symbolic}) = hasmetadata(x, UncertainParameter)
