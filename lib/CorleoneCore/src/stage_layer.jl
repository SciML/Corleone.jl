struct SimulationStage{N, PTYPE, P, G, TSTOPS, SAVEAT} <: AbstractTimeGridLayer{TSTOPS, SAVEAT} 
    "The name of the stage"
    name::N
    "The problem constructor"
    problem::P
    "The simulation grid"
    grid::G
end

function SimulationStage(problem::ProblemLayer{PTYPE, <:Any, <:Any, <:Any, TSTOPS, SAVEAT}, grid::G; name = nothing, kwargs...) where {PTYPE, TSTOPS, SAVEAT, G}
    SimulationStage{typeof(name), PTYPE, typeof(problem), G, TSTOPS, SAVEAT}(name, problem, grid)
end

# Mega Constructor
function SimulationStage(t::Type{T}, model, initials, grid::Pair...; name = nothing, solver::DEAlgorithm, ensemble_algorithm::SciMLBase.EnsembleAlgorithm = EnsembleSerial(), kwargs...) where T <: SciMLBase.DEProblem
    problem = ProblemLayer(t, model, initials; kwargs...)
    grid = SimulationGrid(grid...; solver, ensemble_algorithm)
    return SimulationStage(problem, grid; name, kwargs...)
end

function ODEStage(args...; inplace::Bool, kwargs...)
    SimulationStage(ODEProblem{inplace}, args...; kwargs...)
end

function DAEStage(args...; inplace::Bool, kwargs...) 
    SimulationStage(DAEProblem{inplace}, args...; kwargs...)
end


LuxCore.initialparameters(rng::Random.AbstractRNG, layer::SimulationStage) = (; problem = LuxCore.initialparameters(rng, layer.problem), grid = LuxCore.initialparameters(rng, layer.grid))
LuxCore.parameterlength(layer::SimulationStage) = LuxCore.parameterlength(layer.model) + LuxCore.parameterlength(layer.grid)
LuxCore.initialstates(rng::Random.AbstractRNG, layer::SimulationStage) = (; problem = LuxCore.initialstates(rng, layer.problem), grid = LuxCore.initialstates(rng, layer.grid))

function (s::SimulationStage)(inits, ps, st) 
    prob, problem_st = s.problem(inits, ps.problem, st.problem)
    sols, grid_st = s.grid(prob, ps.grid, st.grid)
    return sols, (; problem = problem_st, grid = grid_st)
end