module CorleoneModelingToolkitExtension

@info "Loading MTK Extension"

using Corleone
using ModelingToolkit

using Corleone.LuxCore
using Corleone.Random
using Corleone.DocStringExtensions
using Corleone.SciMLBase

using ModelingToolkit.Symbolics
using ModelingToolkit.SymbolicUtils
using ModelingToolkit.Setfield
using ModelingToolkit.SymbolicIndexingInterface
using ModelingToolkit.Symbolics.RuntimeGeneratedFunctions

using ModelingToolkit: IndexCache, DiscreteIndex, BufferTemplate, ParameterTimeseriesIndex


RuntimeGeneratedFunctions.init(@__MODULE__)

function _control_timeseries_pairs(controls)
    pairs = Pair[]
    for (i, c) in enumerate(values(controls.controls))
        name = c.name
        # Whole control timeseries, e.g. `u(t)`.
        push!(pairs, name => ParameterTimeseriesIndex(i, :))
        # Per-component access for array-valued controls, e.g. `u[1]`.
        try
            for j in eachindex(name)
                push!(pairs, name[j] => ParameterTimeseriesIndex(i, j)
                )
            end
        catch
            nothing
        end
    end
    return pairs
end

function _mtk_parameter_index_map(sys, controls)
    syms = collect(parameter_symbols(sys))

    # Add component symbols for array-valued parameters so `traj.ps[α[1]]` works.
    for sym in copy(syms)
        try
            for j in eachindex(sym)
                subsym = sym[j]
                if !any(isequal(subsym), syms)
                    push!(syms, subsym)
                end
            end
        catch
            nothing
        end
    end

    for c in values(controls.controls)
        name = c.name
        if !any(isequal(name), syms)
            push!(syms, name)
        end
        try
            for j in eachindex(name)
                sym = name[j]
                if !any(isequal(sym), syms)
                    push!(syms, sym)
                end
            end
        catch
            nothing
        end
    end

    idxmap = Dict{Any, Any}()
    for sym in syms
        idxmap[sym] = parameter_index(sys, sym)
    end
    return idxmap
end

function Corleone.remake_system(sys::ModelingToolkit.AbstractSystem, controls)
    params = _mtk_parameter_index_map(sys, controls)
    return SymbolCache(
        variable_symbols(sys),
        params,
        independent_variable_symbols(sys);
        timeseries_parameters = Dict(_control_timeseries_pairs(controls)),
    )
end

#= 
include("MTKExtension/utils.jl")

include("MTKExtension/optimal_control.jl")
 =#
end
