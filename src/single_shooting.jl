"""
$(TYPEDEF)
Defines a callable layer that integrates the `AbstractDEProblem` `problem` using the specified
`algorithm`. Controls are assumed to impact differential equation via its parameters `problem.p`
at the positions indicated via `control_indices` and are itself specified via `controls`.
Moreover, initial conditions `problem.u0` that are degrees of freedom to be optimized can be
specified by their indices via `tunable_ic` along with their upper and lower bounds via `bounds_ic`.

# Fields
$(FIELDS)

Note: The orders of both `controls` and `control_indices`, and `bounds_ic` and `tunable_ic`
are assumed to be identical!
"""
struct SingleShootingLayer{A,C,U} <: LuxCore.AbstractLuxContainerLayer{(:controls, :problem_remaker)}
    "The initial problem remaker"
    problem_remaker::U
    "The algorithm with which `problem` is integrated."
    algorithm::A
    "The controls"
    controls::C
end

function init_problem(prob, alg)
    return remake_problem(prob, SciMLBase.init(prob, alg))
end

function remake_problem(prob::ODEProblem, state)
    return remake(prob; u0=state.u)
end

function remake_problem(prob::DAEProblem, state)
    return remake(prob; u0=state.u, du0=state.du)
end

"""
$(SIGNATURES)

Constructs a SingleShootingLayer from an `AbstractDEProblem` and a suitable `AbstractDEAlgorithm`
`alg`.

# Arguments
    - `control_indices` : Vector of indices of `prob.p` that denote controls
    - `controls`: Tuple of `ControlParameter` specifying the controls
    - `tunable_ic`: Vector of indices of `prob.u0` that is tunable, i.e., a degree of freedom
    - `bounds_ic` : Vector of tuples of lower and upper bounds of tunable initial conditions
"""
function SingleShootingLayer(
    prob::SciMLBase.AbstractDEProblem,
    alg::SciMLBase.AbstractDEAlgorithm;
    controls=(),
    kwargs...,
)

    # We check if the problem defines a symbolic cache, otherwise we build one from the problem definition 
    psyms = parameter_symbols(prob)
    vsyms = variable_symbols(prob)

    u0syms = get(kwargs, :tunable_u0, ())
    tunable_psyms = get(kwargs, :tunable_p, ())
    quadrature_indices = get(kwargs, :quadrature_indices, ())

    if !isempty(controls)
        @assert !isempty(psyms) "No parameter symbols could be found in the presence "
        @assert !isempty(setdiff(psyms, Set([c.idx for c in controls]))) "Some controls are not defined as parameters."
    end

    if !isempty(tunable_psyms)
        @assert !isempty(psyms) "No parameters symbols could be found in the presence of tunable parameters."
        @assert !isempty(setdiff(psyms, Set(tunable_psyms))) "Some parameter symbols are not defined as parameters."
    end

    if !isempty(u0syms)
        @assert !isempty(vsyms) "No variable symbols could be found in the presence of tunable initial conditions."
        @assert !isempty(setdiff(vsyms, Set(u0syms))) "Some initial conditions are not defined as variables"
    end

    if !isempty(quadrature_indices)
        @assert !isempty(vsyms) "No variable symbols could be found in the presence of quadrature indices."
        @assert !isempty(setdiff(vsyms, Set(quadrature_indices))) "Some quadrature indices are not defined as variables."
    end

    _prob = init_problem(remake(prob), alg)

    controls = NamedTuple(map(controls) do control
        nameof(control) => control
    end)

    rmk = ProblemRemaker(_prob; kwargs...)

    return SingleShootingLayer{typeof(alg),typeof(controls),typeof(rmk)}(
        rmk,
        alg,
        controls,
    )
end

get_quadrature_indices(layer::SingleShootingLayer) = get_quadrature_indices(layer.problem_remaker)
is_shooting_layer(layer::SingleShootingLayer) = true
get_tunable_u0(layer::SingleShootingLayer) = get_tunable_u0(layer.problem_remaker)

function SciMLBase.remake(layer::SingleShootingLayer; kwargs...)
    problem_remaker = remake(layer.problem_remaker; kwargs...)
    algorithm = get(kwargs, :algorithm, layer.algorithm)
    controls = map(layer.controls) do control
        remake(control; kwargs...)
    end
    SingleShootingLayer{typeof(algorithm),typeof(controls),typeof(problem_remaker),}(
        problem_remaker,
        algorithm,
        controls,
    )
end

get_bounds(layer::SingleShootingLayer) = ((;
        controls=map(get_lower_bound, layer.controls),
        problem_remaker=get_lower_bound(layer.problem_remaker),
    ), (;
        controls=map(get_upper_bound, layer.controls),
        problem_remaker=get_upper_bound(layer.problem_remaker),
    ))

function LuxCore.initialstates(rng::Random.AbstractRNG, layer::SingleShootingLayer)
    (; controls, problem_remaker) = layer
    (; problem) = problem_remaker
    control_states = map(Base.Fix1(LuxCore.initialstates, rng), controls)
    initial_states = LuxCore.initialstates(rng, problem_remaker)
    t = reduce(vcat, map(control_states) do cs
        deepcopy(cs.t)
    end)
    append!(t, collect(problem.tspan))
    sort!(t)
    unique!(t)

     sys = symbolic_container(problem)
    timeseries_parameters = map(enumerate(controls)) do (i, k)
         k.idx => ParameterTimeseriesIndex(i, 1)
    end

    newsys = SymbolCache(
        variable_symbols(sys),
        parameter_symbols(sys),
        independent_variable_symbols(sys);
        timeseries_parameters = Dict(timeseries_parameters...)
      )

    control_indices = ntuple(i -> controls[i].idx, length(controls))
    (; controls=control_states, problem_remaker=initial_states, 
    timestops=Tuple(t), control_indices=control_indices, sys = newsys)
end

@generated function _eval_controls(controls::NamedTuple{fields}, t, ps, st) where {fields}
    returns = [gensym() for _ in fields]
    rt_states = [gensym() for _ in fields]
    expr = Expr[]
    for (i, sym) in enumerate(fields)
        push!(expr, :(($(returns[i]), $(rt_states[i])) = controls.$(sym)(t, ps.$(sym), st.$(sym))))
    end
    push!(expr,
        :(st = NamedTuple{$fields}((($(Tuple(rt_states)...),))))
    )
    push!(expr, :(result = ($(returns...),)))
    push!(expr, :(return result, st))
    return Expr(:block, expr...)
end

function (layer::SingleShootingLayer)(x, ps, st::NamedTuple{fields}) where {fields}
    (; problem_remaker, algorithm, controls) = layer
    (; timestops) = st
    problem, problem_st = problem_remaker(x, ps.problem_remaker, st.problem_remaker)
    solutions, _ = eval_problem(problem, algorithm, controls, ps, st, timestops)
    return Trajectory(layer, solutions...; 
        control_parameters = ps.controls, control_states = st.controls, 
        sys = st.sys,
        ), merge(st, (; problem_remaker=problem_st))
end

@generated function eval_problem(prob, algorithm, controls, ps, st, timestops::NTuple{N,<:Real}) where N
    partions = vcat(collect(1:MAXBINSIZE:N), N)
    unique!(partions)
    sols = [gensym() for _ in 1:(length(partions)-1)]
    expr = Expr[]
    retex = Expr(:tuple)
    for i in 1:length(partions)-1
        start = partions[i]
        stop = partions[i+1]
        push!(expr, :($(sols[i]) = _eval_problem(prob, algorithm, controls, ps, st, timestops[$(start):$(stop)])))
        push!(expr, :(prob = remake(prob, u0=last($(sols[i])).u[end])))
        push!(retex.args, Expr(:..., sols[i]))
    end
    push!(expr, :(return $retex, st))
    return Expr(:block, expr...)
end

@generated function _eval_problem(problem, algorithm, controls, ps, st, timestops::NTuple{N,<:Real}) where N 
    sols  = [gensym() for _ in Base.OneTo(N-1)]
    sts = [gensym() for _ in Base.OneTo(N-1)]
    csym = gensym()
    psym = gensym()
    exprs = Expr[] 
    for i in Base.OneTo(N-1)
         push!(exprs, :(($(csym), $(sts[i])) = _eval_controls(controls, timestops[$i], ps.controls, st.controls))) 
         push!(exprs, :($psym = __remake_wrap(problem, problem.p, collect(st.control_indices), collect($csym))))
         push!(exprs, :($(sols[i]) = solve(problem, algorithm, p=$psym, tspan = (timestops[$i], timestops[$(i+1)])))) 
         push!(exprs, :(problem = remake(problem, u0=$(sols[i]).u[end])))
    end
    push!(exprs, Expr(:tuple, sols...))
    return Expr(:block, exprs...)
end


"""
$(SIGNATURES)

Compute the block structure of the hessian of the Lagrangian of an optimal control problem.
As this is a `SingleShootingLayer`, this hessian is dense. See also [``MultipleShootingLayer``](@ref).
"""
function get_block_structure(
    layer::SingleShootingLayer
)
    return vcat(0, LuxCore.parameterlength(layer))
end


get_problem(layer::SingleShootingLayer) = get_problem(layer.problem_remaker)
get_tspan(layer::SingleShootingLayer) = get_tspan(layer.problem_remaker)


function Trajectory(::SingleShootingLayer, sol...; 
    sys, 
    control_parameters = NamedTuple(),
    control_states::NamedTuple = NamedTuple(),
    kwargs...
    )
    u = reduce(vcat, map(sol) do s
        maybevec(s.u)
    end)
    p = first(sol).prob.p
    t = reduce(vcat, map(sol) do s
       s.t
    end)
   
    tseries = map(keys(control_parameters)) do k 
        DiffEqArray(
            maybevec(getproperty(control_parameters, k).u),
            getfield(control_states, k).t,
        )
    end
    controls = ParameterTimeseriesCollection(tseries, deepcopy(p))
    Trajectory{typeof(sys), typeof(u), typeof(p), typeof(t), typeof(controls), Nothing}(sys, u, p, t, controls, nothing)
end
