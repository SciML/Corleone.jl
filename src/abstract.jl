function Base.show(io::IO, layer::OEDLayer)
    fixed = is_fixed(layer)
    discrete = is_discrete(layer)

    dims = layer.dimensions
    type_color, no_color = SciMLBase.get_colorizers(io)

    print(io,
        type_color, "OEDLayer ",
        no_color,  "with $(dims.nh) observation functions and $(dims.np_fisher) considered parameters.\n",
        "Discrete measurements: ",
        type_color, "$discrete. ",
        no_color, "States and sensitities fixed: ",
        type_color, "$fixed.\n")
    print(io, no_color, "Underlying problem: ")
    Base.show(io, "text/plain", remake(get_problem(layer.layer), tspan=get_tspan(layer.layer)))
end

function Base.show(io::IO, layer::MultiExperimentLayer{<:Any, OEDLayer})
    type_color, no_color = SciMLBase.get_colorizers(io)

    print(io,
        type_color, "MultiExperimentLayer ",
        no_color,  "with $(layer.n_exp) experiments to be designed.\n")
    print(io, no_color, "Underlying ",
        type_color,  "OEDLayer: ")
    Base.show(io, "text/plain", layer.layers)
end

function Base.show(io::IO, layer::MultiExperimentLayer{<:Any, <:Tuple})
    type_color, no_color = SciMLBase.get_colorizers(io)

    print(io,
        type_color, "MultiExperimentLayer ",
        no_color,  "with $(layer.n_exp) experiments to be designed.\n")
end
