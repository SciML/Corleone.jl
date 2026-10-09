module CorleoneBase

using SciMLBase: SciMLBase, remake
using CommonSolve: CommonSolve, solve
import ArrayInterface

include("abstractsequential.jl")
include("abstractparallel.jl")

include("solution.jl")

include("sequential.jl")
# Re-export types for user convenience
export SequentialProblem, SolutionWrapper


end # module CorleoneBase
