"""
$(TYPEDEF)

Defines a simulation grid for a single `<:DEProblem` and simulates it with different initial 
conditions specified in the states and parameters of the layer. In general, this can be used to 
implement a shooting grid, different instances of the same problem. 

Assumes that the input in form of the `<:DEProblem` is already modified.

# Fields
$(FIELDS)
"""
struct SimulationGrid{N,S,T<:NamedTuple,U<:NamedTuple,K,E} <: AbstractTimeGridLayer{false,false}
    "The name of the layer"
    name::N
    "The integrator"
    solver::S
    "The associated time spans"
    tspans::T
    "The associated initial conditions"
    initial_conditions::U
    "Additional solver kwargs"
    kwargs::K
    "The ensemble algorithm"
    ensemble_algorithm::E
end

LuxCore.initialparameters(::Random.AbstractRNG, layer::SimulationGrid) = deepcopy(layer.initial_conditions)
LuxCore.parameterlength(layer::SimulationGrid) = LuxCore.parameterlength(layer.initial_conditions)
LuxCore.initialstates(::Random.AbstractRNG, layer::SimulationGrid) = NamedTuple{(:solver, :tspans, :kwargs, :ensemble_algorithm)}((layer.solver, layer.tspans, layer.kwargs, layer.ensemble_algorithm))

# TODO Maybe add more tests here?
# Big TODO: Add tests for initial conditions. Eg. size, eltype etc.

function check_tspans(tspans)
    @assert all(Base.Fix2(<:, Tuple{T,T} where {T<:Number}), typeof.(tspans)) "All time spans must be given as a Tuple{T, T}."
end

function SimulationGrid(grid::Pair...; name=nothing,
    solver::DEAlgorithm,
    ensemble_algorithm::SciMLBase.EnsembleAlgorithm=EnsembleSerial(),
    kwargs...
)
    # Assert the timespans
    check_tspans(first.(grid))

    # Recover the grid 
    sortings = sortperm(collect(first.(grid)), by=first) # We sort by t_0
    # Generate names
    gridnames = ntuple(i -> Symbol(:shooting_interval_, i), size(grid, 1))
    tspans = NamedTuple{gridnames}((first.(grid)[sortings]...,))
    initial_conditions = NamedTuple{gridnames}((last.(grid)[sortings]...,))
    kwargs = NamedTuple(kwargs)

    return SimulationGrid{typeof(name),typeof(solver),typeof(tspans),typeof(initial_conditions),typeof(kwargs),typeof(ensemble_algorithm)}(
        name, solver, tspans, initial_conditions, kwargs, ensemble_algorithm
    )
end

(g::SimulationGrid)(::Any, ps, st::NamedTuple) = throw(MethodError(g, "The simulation grid layer can only be used with a <: DEProblem."))

# Apply 
function (::SimulationGrid)(prob::DEProblem, ps, st::NamedTuple)
    (; tspans, solver, kwargs, ensemble_algorithm) = st
    # Check if all names are equal 
    @assert isempty(setdiff(keys(tspans), keys(ps))) "The provided initial conditions and time spans do not have the same keys!"

    names = keys(tspans)

    # Build the remaker 
    remaker = let ps = ps, tspans = tspans, names = names
        (_problem, i, repeat) -> begin
            int_id = names[i]
            tspan = getproperty(tspans, int_id)
            ics = getproperty(ps, int_id)
            remake(_problem; tspan=tspan, ics...)
        end
    end

    ensembleprob = EnsembleProblem(
        prob, prob_func=remaker
    )

    solutions = solve(ensembleprob, solver, ensemble_algorithm; trajectories=length(tspans), kwargs...)
    return solutions, st
end

