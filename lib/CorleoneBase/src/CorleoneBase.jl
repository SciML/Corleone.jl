module CorleoneBase

using SciMLBase
using CommonSolve

using SymbolicIndexingInterface

using DocStringExtensions

using Random

include("abstractproblem.jl")

include("sequential.jl")
# Re-export types for user convenience
export SequentialProblem


end # module CorleoneBase