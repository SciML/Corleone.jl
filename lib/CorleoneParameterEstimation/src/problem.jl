"""
    CorleoneParameterEstimationProblem <: SciMLBase.AbstractNonlinearProblem

A lightweight container tying a [`ParameterEstimationLayer`](@ref) to a flat vector of
optimization variables (the tunable parameters, initial conditions and controls of the wrapped
shooting layer) so that the least-squares residuals can be minimized with any NonlinearSolve
solver.

For a plain parameter-estimation setup (no tunable initial conditions, no controls) the
optimization variables are exactly the tunable model parameters. Parameters that are fixed on
the wrapped layer (e.g. provided via `bounds_p` with coincident lower/upper bounds) are not
part of the optimization variables and are kept at their values.

# Fields
$(FIELDS)

# Constructors
- `CorleoneParameterEstimationProblem(layer; rng, x0)`

The low-level `NonlinearSolve.NonlinearLeastSquaresProblem` can be built directly via
[`problem`](@ref). Most users construct that problem through the convenience constructor
`NonlinearSolve.NonlinearLeastSquaresProblem(pe::ParameterEstimationLayer; kwargs...)`.
"""
struct CorleoneParameterEstimationProblem{L, X, B, P, S, R}
    "The wrapped parameter-estimation layer"
    layer::L
    "Structured initial parameters (NamedTuple `(u0, p, controls)`)"
    ps0::P
    "The layer state"
    st::S
    "Flat vector of optimization variables"
    x0::X
    "Flat lower bounds of the optimization variables (or `nothing`)"
    lb::B
    "Flat upper bounds of the optimization variables (or `nothing`)"
    ub::B
    "In-place residual function `(res, x, p) -> res` reconstructing the structured layer parameters from `x`"
    residual::R
end

# Split sizes for flattening the layer parameter NamedTuple.
_flat_sizes(ps::NamedTuple) = map(ps) do v
    length(v)
end

function _flatten(ps::NamedTuple)
    return reduce(vcat, collect(values(ps)) .|> vec)
end

function _unflatten(ps0::NamedTuple, x::AbstractVector)
    sizes = collect(values(_flat_sizes(ps0)))
    cumul = accumulate(+, sizes)
    starts = cumul .- sizes .+ 1
    vals = [x[s:e] for (s, e) in zip(starts, cumul)]
    return NamedTuple{keys(ps0)}(vals)
end

function CorleoneParameterEstimationProblem(
        layer::ParameterEstimationLayer;
        rng::Random.AbstractRNG = Random.default_rng(),
        x0 = nothing,
    )
    ps0, st = LuxCore.setup(rng, first(get_layer(layer)))
    x0 = isnothing(x0) ? _flatten(ps0) : x0
    lb, ub = _tunable_bounds(layer, ps0)

    residual = let layer = layer, ps0 = ps0, st = st
        (res, x, p) -> begin
            ps = _unflatten(ps0, x)
            return residuals!(res, layer, ps, st)
        end
    end

    return CorleoneParameterEstimationProblem(
        layer, ps0, st, x0, lb, ub, residual
    )
end

# Build flat bounds aligned with the flattened `ps0`. For initial conditions the wrapped layer
# stores full-size u0 bounds while `ps0.u0` only contains the *tunable* entries, so we index by
# the tunable initial-condition indices. Parameters and controls are already sized to match the
# flattened vector by the shooting-layer construction.
function _tunable_bounds(layer::ParameterEstimationLayer, ps0::NamedTuple)
    wrapper = first(get_layer(layer))
    (lb, ub) = Corleone.get_bounds(wrapper)
    flat_lb, flat_ub = eltype(ps0.p)[], eltype(ps0.p)[]

    if haskey(ps0, :u0) && !isempty(ps0.u0)
        idx = get_tunable(wrapper)
        append!(flat_lb, vec(lb.u0[idx]))
        append!(flat_ub, vec(ub.u0[idx]))
    end
    if haskey(ps0, :p)
        append!(flat_lb, vec(lb.p))
        append!(flat_ub, vec(ub.p))
    end
    if haskey(ps0, :controls)
        append!(flat_lb, vec(lb.controls))
        append!(flat_ub, vec(ub.controls))
    end
    return flat_lb, flat_ub
end

"""
    NonlinearSolve.NonlinearLeastSquaresProblem(problem::CorleoneParameterEstimationProblem)

Build the `NonlinearLeastSquaresProblem` (with optional `lb`/`ub`) driving the least-squares
fit of the layer's tunable parameters.
"""
function SciMLBase.NonlinearLeastSquaresProblem(problem::CorleoneParameterEstimationProblem)
    (; residual, x0, lb, ub) = problem
    n_obs = n_observations(get_data(problem.layer))
    # Out-of-place residual so NonlinearSolve can accurately infer the residual dimension
    # (number of observations), which generally differs from the number of fitted parameters.
    residual_oop = (x, p) -> begin
        r = similar(x, n_obs)
        residual(r, x, p)
        return r
    end
    return NonlinearLeastSquaresProblem(
        NonlinearFunction{false}(residual_oop), x0; lb = lb, ub = ub
    )
end

"""
    NonlinearSolve.NonlinearLeastSquaresProblem(pe::ParameterEstimationLayer; kwargs...)

Construct the `NonlinearLeastSquaresProblem` for a [`ParameterEstimationLayer`](@ref) so it
can be solved with a NonlinearSolve least-squares solver, e.g.
`solve(prob, GaussNewton())` or `solve(prob, LevenbergMarquardt())`. Additional keywords are
forwarded to [`CorleoneParameterEstimationProblem`](@ref).
"""
function SciMLBase.NonlinearLeastSquaresProblem(
        pe::ParameterEstimationLayer; kwargs...
    )
    return NonlinearLeastSquaresProblem(CorleoneParameterEstimationProblem(pe; kwargs...))
end

"""
    problem(pe::ParameterEstimationLayer; kwargs...)

Return the `NonlinearSolve.NonlinearLeastSquaresProblem` for `pe` (an alias for
`NonlinearSolve.NonlinearLeastSquaresProblem(pe; kwargs...)`).
"""
problem(pe::ParameterEstimationLayer; kwargs...) =
    NonlinearLeastSquaresProblem(pe; kwargs...)
