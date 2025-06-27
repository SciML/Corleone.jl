"""
$(TYPEDEF)

Struct to hold measurement information gain and corresponding multipliers
"""
struct InformationGain{L,G,T,M}
    "Local information gain"
    LIG::L
    "Global information gain"
    GIG::G
    "Timepoints"
    t::T
    "Multiplier corresponding to measurement constraints"
    μ::M
end

function Corleone.InformationGain(builder::OEDProblemBuilder, predictor::OCPredictor, u_opt; kwargs...)
    fwdsol = predictor(u_opt; kwargs...)[1];


    ob = map(x -> x.rhs, ModelingToolkit.observed(builder.system))

    sts = filter(x -> Corleone.is_statevar(x) && !Corleone.is_fim(x) && !Corleone.is_sensitivity(x), unknowns(builder.system))
    G_states = filter(Corleone.is_sensitivity, unknowns(builder.system))

    nx = length(sts)
    np = Int(length(G_states)/nx)

    G_states = map(Iterators.product(Base.OneTo(2), Base.OneTo(np))) do (i,j)
        ModelingToolkit.getvar(builder.system, Symbol("G", string(i), string(j)))
    end

    F_states = map(Iterators.product(Base.OneTo(np), Base.OneTo(np))) do (i,j)
        if i <= j
            ModelingToolkit.getvar(builder.system, Symbol("F", string(i), string(j)))
        else
            ModelingToolkit.getvar(builder.system, Symbol("F", string(j), string(i)))
        end
    end


    timepoints = reduce(vcat, map(fwdsol) do sol
        sol.t
    end)

    Pi = map(ob) do obi
        hx = Symbolics.jacobian([obi], sts)
        gram = hx * G_states
        gram' * gram
    end


    Pi_eval = reduce(vcat, map(fwdsol) do sol
        SymbolicIndexingInterface.getsym(sol, Pi)(sol)
    end)

    F_inv = inv(last(getsym(last(fwdsol), F_states)(last(fwdsol))))

    GIG_eval = map(Pi_eval) do Pit
        map(Pit) do Pii
            F_inv' * Pii * F_inv
        end
    end

    multiplier = nothing

    return Corleone.InformationGain{typeof(Pi_eval), typeof(GIG_eval), typeof(timepoints), typeof(multiplier)}(Pi_eval, GIG_eval,
                timepoints, multiplier)
end


