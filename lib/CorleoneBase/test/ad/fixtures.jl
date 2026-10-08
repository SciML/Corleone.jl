abstract type AbstractModelCase{IIP} end

struct LinearCase{IIP} <: AbstractModelCase{IIP} end
struct NonlinearCase{IIP} <: AbstractModelCase{IIP} end

case_name(::LinearCase) = "linear"
case_name(::NonlinearCase) = "nonlinear"

is_inplace(::AbstractModelCase{IIP}) where {IIP} = IIP
form_name(case) = is_inplace(case) ? "in-place" : "out-of-place"

dynamics(::LinearCase{false}) = (u, p, t) -> -p[1] .* u
dynamics(::NonlinearCase{false}) = (u, p, t) -> -p[1] .* u .^ 2

function dynamics(::LinearCase{true})
    return function (du, u, p, t)
        du .= -p[1] .* u
        return nothing
    end
end

function dynamics(::NonlinearCase{true})
    return function (du, u, p, t)
        du .= -p[1] .* u .^ 2
        return nothing
    end
end

const CASES = (
    LinearCase{false}(), LinearCase{true}(),
    NonlinearCase{false}(), NonlinearCase{true}(),
)
const INPUTS = ([1.2, 0.4, 0.7, 0.8], [0.9, 0.25, 0.55, 1.1])
const WEIGHTS = (1.0, 2.0, 3.0)
const TARGETS = (0.1, 0.2, 0.3)
const SOLVE_KWARGS = (
    abstol = 1.0e-11, reltol = 1.0e-11,
    save_everystep = false, save_start = true, save_end = true, dense = false,
)

function sequential_solutions(case, x; solve_kwargs = (;))
    # Deliberately different from x: stage one must use solve-time overrides.
    template = ODEProblem(dynamics(case), [9.0], (0.0, 1.0), [4.0])
    transition = function (sol, next_stage)
        if next_stage == 2
            return remake(sol.prob; u0 = sol.u[end], p = [x[3]], tspan = (1.0, 2.0))
        elseif next_stage == 3
            return remake(sol.prob; u0 = [x[4]], tspan = (2.0, 3.0))
        end
        throw(ArgumentError("Unexpected stage $next_stage"))
    end
    problem = SequentialProblem(template; transition, terminal = (sol, i) -> i >= 3)
    options = merge(SOLVE_KWARGS, solve_kwargs)
    result = CommonSolve.solve(problem, Tsit5(); options..., u0 = [x[1]], p = [x[2]])
    # The current full solve returns the completed iterator, which owns the buffer.
    return result.buffer
end

endpoint_loss(y) = sum(WEIGHTS[i] * (y[i] - TARGETS[i])^2 for i in 1:3) / 2

function sequential_loss(case, x; solve_kwargs = (;))
    solutions = sequential_solutions(case, x; solve_kwargs)
    return endpoint_loss(ntuple(i -> solutions[i].u[end][1], 3))
end

# Independent closed-form endpoints and hand-derived Jacobians. No AD or
# numerical solve is used to compute the reference gradients.
function reference_endpoints(::LinearCase, x)
    u1, p1, p2, u3 = x
    e1, e2 = exp(-p1), exp(-p2)
    y = (u1 * e1, u1 * e1 * e2, u3 * e2)
    jacobian = [
        e1 -y[1] 0.0 0.0
        e1 * e2 -y[2] -y[2] 0.0
        0.0 0.0 -y[3] e2
    ]
    return y, jacobian
end

function reference_endpoints(::NonlinearCase, x)
    u1, p1, p2, u3 = x
    d1, d2, d3 = 1 + p1 * u1, 1 + (p1 + p2) * u1, 1 + p2 * u3
    y = (u1 / d1, u1 / d2, u3 / d3)
    jacobian = [
        inv(d1^2) -y[1]^2 0.0 0.0
        inv(d2^2) -y[2]^2 -y[2]^2 0.0
        0.0 0.0 -y[3]^2 inv(d3^2)
    ]
    return y, jacobian
end

function reference_value_and_gradient(case, x)
    y, jacobian = reference_endpoints(case, x)
    gradient = [
        sum(jacobian[i, j] * WEIGHTS[i] * (y[i] - TARGETS[i]) for i in 1:3)
            for j in eachindex(x)
    ]
    return endpoint_loss(y), gradient
end
