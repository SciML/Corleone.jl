"""
$(TYPEDEF)

This function is a structured model for dynamics. It should be handled with care. 
In general, we assume that the `structure` is a function which maps a `NamedTuple` to the 
corresponding outputs for the underlying `DEFunction`. 

The structure is stateless. All components are assumed to be layers.

A simple example is the the universal differential equation. We want to use the dynamic model to 
TODO Need to finish writing here.

```julia 
    # Define the components
    neural_network = Chain((du,u,t)->u, Dense(...), ...) # Returns another vector
    prior_knowledge = WrappedFunction((du,u,t) -> u) # Simply returns u 
    # Define the structure
    structure = (du, u, t, nt::NamedTuple) -> begin 
        u_known = nt.prior_knowledge 
        u_unknown = nt.neural_network 
        du .= u_known .+ u_unkown
        return du  # Same as the ODE Function
    end

    # Define the model 
    # Takes in (du, u, t) and returns du
    DynamicModel(:MyUDE, structure, (; neural_network, prior_knowledge))
```

# Fields
$(FIELDS)
"""
struct DynamicModel{N,S,C} <: LuxCore.AbstractLuxContainerLayer{(:components,)}
    "The name of the model"
    name::N 
    "The model structure"
    structure::S 
    "The individual components"
    components::C
end

function (d::DynamicModel)(input, ps, st::NamedTuple)
    output, component_states = apply_components(d.components, input, ps.components, st.components)
    out = d.structure(input..., output)
    return out , merge(st, (; components = component_states))
end

@generated function apply_components(components::NamedTuple{fields}, input, ps, st::NamedTuple{fields}) where fields 
    N = length(fields)
    outsyms = [gensym() for _ in 1:N]
    stsyms = [gensym() for _ in 1:N]
    exprs = Expr[]
    for i in 1:N 
        fname = fields[i]
        push!(exprs, 
            :(($(outsyms[i]), $(stsyms[i])) = components.$(fname)(input, ps.$(fname), st.$(fname)))
        )
    end
    push!(exprs, :(new_st = NamedTuple{$(fields)}(($(stsyms...),))))
    push!(exprs, :(returns = NamedTuple{$(fields)}(($(outsyms...),))))
    push!(exprs, :(return returns, new_st))
    return Expr(:block, exprs...)
end

