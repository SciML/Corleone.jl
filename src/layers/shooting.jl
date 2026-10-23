@concrete terse struct ShootingLayer <: LuxCore.AbstractLuxContainerLayer{(:initialcondition, :parameter)}
    name
    problem
    alg
    ensemblealg
    initialcondition
    parameter
    sys
end

# Resolve symbolic metadata once, outside the forward/AD path. Keep the existing
# positional constructor for callers that replace the problem or sublayers.
ShootingLayer(name, problem, alg, ensemblealg, initialcondition, parameter) =
    ShootingLayer(name, problem, alg, ensemblealg, initialcondition, parameter, get_symbolic_container(problem))


function ShootingLayer(
        problem::SciMLBase.AbstractSciMLProblem,
        alg::SciMLBase.AbstractSciMLAlgorithm,
        definitions...;
        name = nothing,
        fixed_parameters = [],
        ensemblealg = EnsembleSerial(),
    )
    definitions = collect(definitions)
    ic_defs = findall(definitions) do definition
        annotation = definition isa Pair ? first(definition) : definition
        annotation isa AbstractAnnotation && is_variable(annotation)
    end
    p_defs = setdiff(eachindex(definitions), ic_defs)
    ics = InitialConditions(problem, definitions[ic_defs]...)
    params = Parameter(problem, definitions[p_defs]...; fixed = fixed_parameters)
    return ShootingLayer(name, problem, alg, ensemblealg, ics, params)
end

function get_timestops(layer::ShootingLayer, ps, st, T = eltype(layer.problem.tspan))
    tstops = vcat(
        collect(T, layer.problem.tspan),
        get_timestops(layer.initialcondition, ps.initialcondition, st.initialcondition, T),
        get_timestops(layer.parameter, ps.parameter, st.parameter, T)
    )
    filter!(t -> first(layer.problem.tspan) <= t <= last(layer.problem.tspan), tstops)
    sort!(tstops)
    return unique!(tstops)
end

(x::ShootingLayer)(::Any, ps, st) = x(x.problem, ps, st)

@inline function (x::ShootingLayer)(prob::SciMLBase.AbstractSciMLProblem, ps, st::NamedTuple)
    tstops = get_timestops(x, ps, st)
    return x((prob, tstops), ps, st)
end

@inline function (x::ShootingLayer)((prob, tstops)::Tuple, ps, st::NamedTuple)
    solutions, residuals, next_st = solve_problems(x.ensemblealg, x, prob, ps, st, tstops)
    return build_trajectory(x, ps, next_st, solutions, residuals), next_st
end

solve_problems(alg, ::ShootingLayer, prob, ps, st, tstops) = throw(ArgumentError("Unsupported algorithm $(alg). Currently only EnsembleSerial(), EnsembleThreads(), and EnsembleDistributed() are supported."))

@inline function solve_problems(::EnsembleSerial, x::ShootingLayer, prob::SciMLBase.AbstractSciMLProblem, ps, st, tstops)
    t0, tinf = tstops[1:2]
    (u0, variable_residuals), st_ic = x.initialcondition((prob.u0, t0), ps.initialcondition, st.initialcondition)
    p, st_p = x.parameter(t0, ps.parameter, st.parameter)
    sol = solve(prob, x.alg; u0 = u0, p = p, tspan = (t0, tinf), save_everystep = false, dense = false, save_start = true, save_end = true)
    next_st = (; initialcondition = st_ic, parameter = st_p)
    residuals = variable_residuals
    return buffered_solve(x, prob, sol, residuals, ps, next_st, tstops[2:end])
end

get_du0(::Any, ::Any) = nothing
get_du0(::SciMLBase.AbstractDAEProblem, sol) = sol.du[end]

function internal_solve(prob::SciMLBase.AbstractSciMLProblem, alg::SciMLBase.AbstractSciMLAlgorithm, sol::SciMLBase.AbstractSciMLSolution, u0, p, tspan)
    du0 = get_du0(prob, sol)
    return internal_solve(prob, alg, u0, p, tspan, du0)
end

# Keep both sides of segment boundaries; reduce continuous handoffs when assembling the trajectory.
internal_solve(prob, alg, u0, p, tspan, ::Nothing) = solve(prob, alg; u0 = u0, p = p, tspan, save_everystep = false, dense = false, save_start = true, save_end = true)
internal_solve(prob, alg, u0, p, tspan, du0) = solve(prob, alg; u0 = u0, du0 = du0, p = p, tspan, save_everystep = false, dense = false, save_start = true, save_end = true)


function mkbuffer(::T, vec::AbstractArray) where {T}
    buffer = similar(vec, T)
    return is_extension_loaded(Val{:Zygote}()) ? wrapbuffer(buffer) : buffer
end

function wrapbuffer end # Implemented by the Zygote extension.

unwrapbuffer(x) = x

function buffered_solve(x::ShootingLayer, prob, sol::T, res::R, ps, st, tstops) where {T, R}
    solbuff = mkbuffer(sol, tstops)
    solbuff[1] = sol
    resbuff = mkbuffer(res, tstops)
    resbuff[1] = res
    return buffered_solve!(solbuff, x, prob, resbuff, ps, st, tstops)
end

terminal_state(sol::SciMLBase.AbstractTimeseriesSolution) = last(sol.u)
terminal_state(sol::Base.AbstractVecOrTuple) = last(sol)
terminal_state(sol::AbstractArray) = sol[:, end]

@inline function buffered_solve!(sols, x::ShootingLayer, prob, res, ps, st, tstops)
    @inbounds for i in Base.OneTo(length(tstops) - 1)
        (u0, res[i + 1]), st_ic = x.initialcondition((terminal_state(sols[i]), tstops[i]), ps.initialcondition, st.initialcondition)
        p, st_p = x.parameter(tstops[i], ps.parameter, st.parameter)
        sols[i + 1] = internal_solve(prob, x.alg, sols[i], u0, p, (tstops[i], tstops[i + 1]))
        st = (; initialcondition = st_ic, parameter = st_p)
    end
    return unwrapbuffer(sols), unwrapbuffer(res), st
end
