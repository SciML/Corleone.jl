struct InformationGain{T,L,G}
    t::T
    local_information_gain::L
    global_information_gain::G
end


function Corleone.InformationGain(layer::OEDLayer, u_opt; kwargs...)
    ps, st = LuxCore.setup(Random.default_rng(), layer)
    p = ComponentArray(u_opt, getaxes(ComponentArray(ps)))

    sols, _ = layer(nothing, p, st)

    F_tf = begin
        if is_fixed(layer)
            F_vars = [observed_sensitivity_product_variables(layer.layer, i) for i=1:layer.dimensions.nh]
            if isa(sols, EnsembleSolution)
                symmetric_from_vector(sum([last(sols)[F_var][end] for F_var in F_vars]))
            else
                symmetric_from_vector(sum([sols[F_var][end] for F_var in F_vars]))
            end
        else
            fvars = fisher_variables(layer.layer)
            symmetric_from_vector(last(sols)[fvars][end])
        end
    end

    Gvars = sensitivity_variables(layer)

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

    Pi = begin
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

    Fi = begin
        map(Pi) do Ph
            map(Ph) do Ph_ti
                C' * Ph_ti * C
            end
        end
    end


    return InformationGain{typeof(timepoints), typeof(Pi), typeof(Fi)}(timepoints, Pi, Fi)
end
