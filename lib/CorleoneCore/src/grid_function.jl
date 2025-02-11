struct GridFunction{N, S} <: AbstractTimeGridLayer{false, true} 
    "The name of the layer"
    name::N
    "The wrapped initial states"
    initial_states::S
end 


