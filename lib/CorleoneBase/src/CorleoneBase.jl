module CorleoneBase

using SciMLBase: SciMLBase, remake
using CommonSolve: CommonSolve, solve
import ArrayInterface

include("abstractproblem.jl")

include("sequential.jl")
# Re-export types for user convenience
export SequentialProblem


end # module CorleoneBase
