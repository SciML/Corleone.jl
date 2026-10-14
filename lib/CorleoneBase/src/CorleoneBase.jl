module CorleoneBase

using SciMLBase: SciMLBase, remake
using CommonSolve: CommonSolve, solve
# `import` (not `using`) so the module name is bound explicitly; every use is
# qualified as `ArrayInterface.aos_to_soa`, and ExplicitImports rejects the
# implicit module-name import that `using ArrayInterface` introduces.
import ArrayInterface

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
