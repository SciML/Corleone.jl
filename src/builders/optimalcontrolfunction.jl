struct OptimalControlFunction{OOP,IIP,P,C}
    f_oop::OOP
    f_iip::IIP
    predictor::P
    consolidate::C
end

function (f::OptimalControlFunction)(p::AbstractVector{T}, ::Any) where {T}
    (; f_oop, predictor, consolidate) = f
    trajectory, ps = predict(predictor, p)
    f_oop(vec(trajectory), ps, zero(eltype(p))) |> consolidate
end

function (f::OptimalControlFunction)(u::AbstractVector, p::AbstractVector{T}, ::Any) where {T}
    (; f_iip, predictor) = f
    trajectory, ps = predict(predictor, p)
    f_iip(u, vec(trajectory), ps, zero(eltype(p)))
end


function OptimalControlFunction{IIP}(ex, prob, alg::SciMLBase.DEAlgorithm, args...;
    consolidate=identity, tspan=nothing, kwargs...) where {IIP}
    (; system, substitutions) = prob
    t = ModelingToolkit.get_iv(system)
    vars = operation.(ModelingToolkit.unknowns(system))
    empty!(substitutions)
    new_ex = map(ex) do eq
        collect_explicit_timepoints!(substitutions, eq, vars, t)
    end
    statevars, cost_substitutions, saveat = create_cost_substitutions(substitutions, vars)
    new_ex = map(new_ex) do eq
        substitute(eq, cost_substitutions)
    end
    # TODO SPecial case with no simulation needed! 
    _tspan = if isnothing(tspan) && !isempty(saveat)
        extrema(saveat)
    elseif isnothing(tspan) && isempty(saveat)
        # No sim needed 
        (0.0, 0.0)
    else
        tspan
    end
    predictor = OCPredictor{IIP}(system, alg, _tspan, args...; saveat=saveat, kwargs...)
    foop, fiip = generate_custom_function(system, new_ex, vec(statevars[1]); expression=Val{false}, kwargs...)
    return OptimalControlFunction{
        typeof(foop),typeof(fiip),typeof(predictor),typeof(consolidate)
    }(foop, fiip, predictor, consolidate)
end
