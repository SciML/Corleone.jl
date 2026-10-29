include("./lookup.jl")
include("./parameters.jl")

include("./initials.jl")
export InitialConditions

include("./parameter.jl")
export Parameter

include("./shooting.jl")
include("./shooting_parallel.jl")
include("./shooting_initialization.jl")
export forward_initialization, linear_initialization, random_initialization
export constant_initialization, custom_initialization, hybrid_initialization
export ShootingLayer

@non_differentiable get_timestops(layer, ps, st, type)
