module CorleoneBase

using SciMLBase: SciMLBase, remake
using CommonSolve: CommonSolve, solve
using ArrayInterface

include("abstractsequential.jl")
include("abstractparallel.jl")

include("solution.jl")

include("sequential.jl")
include("parallel.jl")
# Re-export types for user convenience
export SequentialProblem 
export ParallelProblem
export SolutionWrapper


end # module CorleoneBase
