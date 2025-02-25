struct ProblemLayer{P, M, I, K, TSTOPS, SAVEAT} <: AbstractTimeGridLayer{TSTOPS, SAVEAT}
    "The underyling model"
    model::M
    "The necessary arguments to construct the problem"
    initials::I
    "Additional keyworded arguments"
    kwargs::K

    function ProblemLayer(::Type{T}, model, initials; kwargs...) where T <: SciMLBase.DEProblem
        TSTOPS = has_tstops(model) || hasproperty(kwargs, :tstops)
        SAVEAT = has_saveats(model) || hasproperty(kwargs, :saveat)
        new{T, typeof(model), typeof(initials), typeof(kwargs), TSTOPS, SAVEAT}(
            model, initials, kwargs
        )
    end
end

LuxCore.initialparameters(rng::Random.AbstractRNG, layer::ProblemLayer) = (; model = LuxCore.initialparameters(rng, layer.model))
LuxCore.parameterlength(layer::ProblemLayer) = LuxCore.parameterlength(layer.model)
LuxCore.initialstates(rng::Random.AbstractRNG, layer::ProblemLayer) = (; model = LuxCore.initialstates(rng, layer.model))

# We assume that inits is a 
function (layer::ProblemLayer{P})(inits, ps, st) where P 
    inits = merge(layer.initials, inits)
    # Wrap the model 
    wrapped_model = InternalWrapper.initialize_model(layer.model, ps.model, st.model)
    problem = P(wrapped_model, values(inits)..., ps.model; layer.kwargs...)
    return problem, st
end



