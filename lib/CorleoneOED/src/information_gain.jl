"""
$(TYPEDEF)
Collects local and global information gain matrices for solutions of OED problems, that
can be used to gain insights into solutions via the problems optimality conditions
and for a-posteriori analyses and visualisations.

# Fields
$(FIELDS)
"""
struct InformationGain{T,L,G}
    "Timepoints"
    t::T
    "Vector of local information gain matrices for different observation functions"
    local_information_gain::L
    "Vector of global information gain matrices for different observation functions"
    global_information_gain::G
end

"""
    InformationGain(oedlayer, u_opt; F=nothing)
Computes local and global information gain matrices for single `OEDLayer` and a solution `u_opt`.
If `F` is supplied, it will be used for the scaling in the global information gain, otherwise
the Fisher information matrix will be calculated from the `u_opt`.
"""
function InformationGain(layer::OEDLayer, u_opt; F=nothing)
    ps, st = LuxCore.setup(Random.default_rng(), layer)
    p = ComponentArray(u_opt, getaxes(ComponentArray(ps)))

    sols, _ = layer(nothing, p, st)

    F_tf = isnothing(F) ? fim(layer, u_opt) : F

    Gvars = Corleone.sensitivity_variables(layer)

    G_states = begin
        if isa(sols, EnsembleSolution)
            reduce(vcat, map(sols) do sol
                sol[Gvars]
            end)
        else
            sols[Gvars]
        end
    end

    timepoints = begin
        if isa(sols, EnsembleSolution)
            reduce(vcat, map(sols) do sol
                sol.t
            end)
        else
            sols.t
        end
    end

    LIG = begin
        if isa(sols, EnsembleSolution)
            map(1:layer.dimensions.nh) do i
                reduce(vcat, map(sols) do sol
                    Gi = sol[Gvars]
                    map(zip(sol, sol.t, Gi)) do (_sol,t, Git)
                        gram = layer.observed.hx(_sol, first(layer.layer.layers).problem.p, t)[i:i,:] * Git
                        gram' * gram
                    end
                end)
            end
        else
            map(1:layer.dimensions.nh) do i
                map(zip(sols, sols.t, G_states)) do (sol,t, Gi)
                    gram = layer.observed.hx(sol, layer.layer.problem.p, t)[i:i,:] * Gi
                    gram' * gram
                end
            end
        end
    end

    C = inv(F_tf)

    GIG = begin
        map(LIG) do Pi
            map(Pi) do Pi_ti
                C' * Pi_ti * C
            end
        end
    end

    return InformationGain{typeof(timepoints), typeof(LIG), typeof(GIG)}(timepoints, LIG, GIG)
end

"""
    InformationGain(multilayer, u_opt)
Computes local and global information gain matrices for all experiments of the
`MultiExperimentlayer` and the solution `u_opt`.
"""
function InformationGain(multilayer::MultiExperimentLayer{<:Any, OEDLayer}, u_opt; kwargs...)
    exp_names = Tuple([Symbol("experiment_$i") for i=1:multilayer.n_exp])
    ps, st = LuxCore.setup(Random.default_rng(), multilayer)
    p = ComponentArray(u_opt, getaxes(ComponentArray(ps)))
    fsym = Corleone.fisher_variables(multilayer.layers)
    sols, _ = multilayer(nothing, p, st)
    F = symmetric_from_vector(sum(map(sols) do sol
        isa(sol, EnsembleSolution) ? last(sol)[fsym][end] : sol[fsym][end]
    end))
    exp_IG = map(1:multilayer.n_exp) do i
        u_local = getproperty(p, Symbol("experiment_$i"))
        InformationGain(multilayer.layers, u_local; F=F)
    end
    return NamedTuple{exp_names}(exp_IG)
end

function InformationGain(multilayer::MultiExperimentLayer{<:Any, <:Tuple}, u_opt; kwargs...)
    exp_names = Tuple([Symbol("experiment_$i") for i=1:multilayer.n_exp])
    ps, st = LuxCore.setup(Random.default_rng(), multilayer)
    p = ComponentArray(u_opt, getaxes(ComponentArray(ps)))
    sols, _ = multilayer(nothing, p, st)
    F = symmetric_from_vector(sum(map(zip(sols, multilayer.layers)) do (sol, layer)
        fsym = Corleone.fisher_variables(layer)
        isa(sol, EnsembleSolution) ? last(sol)[fsym][end] : sol[fsym][end]
    end))

    exp_IG = map(multilayer.layers) do layer
        u_local = getproperty(p, Symbol("experiment_$i"))
        InformationGain(layer, u_local; F=F)
    end
    return NamedTuple{exp_names}(exp_IG)
end
