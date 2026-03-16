"""
$(TYPEDEF)

A struct for capturing the internal definition of a dynamic optimization problem.

# Fields
$(FIELDS)
"""
struct CorleoneDynamicOptProblem{L,CB} <: LuxCore.AbstractLuxWrapperLayer{:layer}
	"The [`ObservedLayer`](@ref) resulting"
	layer::L 
    "Lower bounds for the constraints"
    lcons::CB
    "Upper bounds for the constraints"
    ucons::CB
end

function normalize_constraint(expr::Expr, ::Type{T}) where T
	@assert expr.head == :call "The expression is not a call."
	op, a, b = expr.args
if op == :(<=)
		Expr(:call, :(-), a, b), T(-Inf), zero(T)  
elseif op == :(>=)
		Expr(:call, :(-), b, a), T(-Inf), zero(T)  
elseif op == :(==)
		Expr(:call, :(-), a, b), zero(T), zero(T)
	else 
		throw(error("The operator $(op) is not suppported to define constraints. Only ==, <=, and >= are supported."))
	end
end 

function CorleoneDynamicOptProblem(layer::LuxCore.AbstractLuxLayer, objective::Expr, constraints::Expr...; 
								   kwargs...
	) 
	problem = get_problem(layer) 
	T = eltype(problem.u0) 
	
	lb = fill(zero(T), length(constraints) + get_number_of_shooting_constraints(layer)) 
	ub = fill(zero(T), length(constraints) + get_number_of_shooting_constraints(layer))
	constraints = map(enumerate(constraints)) do (i, con) 
		con, lb[i], ub[i] = normalize_constraint(con, T)
		con 
	end
	observed = ObservedLayer(layer, objective, constraints...; kwargs...)  
	CorleoneDynamicOptProblem{typeof(observed), typeof(lb)}(observed, lb, ub)
end

# A simple wrapper for reconstructing the parameters 

abstract type AbstractVectorizer end 

struct WrappedFunction{F, PR, PO} 
	f::F 
	pre::PR 
	post::PO
end 

(f::WrappedFunction)(u, p) = f.post(f.f(nothing, f.pre(u), p)) 
(f::WrappedFunction)(res, u, p) = f.post(res, f.f(nothing, f.pre(u), p)) 

function WrappedFunction(::Any, f, p, st; post = identity, kwargs...) end

function to_vec(::Any, p) end

"""
$(SIGNATURES)

Construct a SciML `OptimizationFunction` from a [`CorleoneDynamicOptProblem`](@ref).
"""
function SciMLBase.OptimizationFunction(
    prob::CorleoneDynamicOptProblem, ad::SciMLBase.ADTypes.AbstractADType;
	vectorizer, 
    rng::Random.AbstractRNG=Random.default_rng(),
    kwargs...
)
	p, st = LuxCore.setup(rng, prob.layer)
	n_shoot = get_number_of_shooting_constraints(prob.layer.layer) 
	objective = WrappedFunction(vectorizer, prob.layer, p, st; post = x -> first(x) )
	cons = WrappedFunction(vectorizer, prob.layer, p, st; post = let n_shoot = n_shoot
		(res, (x,_...)) -> begin
			for i in Base.OneTo(n_shoot) 
				res[i] = x.observed[i+1]
			end
			shooting_constraints!(res[n_shoot+1:end], x.trajectory) 	
			return res 
		end
		end)
    return SciMLBase.OptimizationFunction{true}(objective, ad; cons, kwargs...)
end

"""
$(SIGNATURES)

Construct a SciML `OptimizationProblem` from a [`CorleoneDynamicOptProblem`](@ref).
"""
function SciMLBase.OptimizationProblem(
    prob::CorleoneDynamicOptProblem, ad::SciMLBase.ADTypes.AbstractADType;
	vectorizer, 
    rng::Random.AbstractRNG=Random.default_rng(),
    sense=nothing,
    kwargs...
)
	p, st = LuxCore.setup(rng, prob)
	n_shoot = get_number_of_shooting_constraints(prob.layer.layer) 
	objective = WrappedFunction(vectorizer, prob, p, st; post = x -> first(x).observations[1])
	cons = WrappedFunction(vectorizer, prob, p, st; post = let n_shoot = n_shoot, n_cons = size(prob.lcons, 1)
		(res, x) -> begin
			x = first(x)
			shooting_constraints!(res[1:n_shoot], x.trajectory) 	
			for i in Base.OneTo(n_cons) 
				res[n_shoot + i] = x.observations[i+1]
			end
			return res 
		end
		end)
    optf = SciMLBase.OptimizationFunction{true}(objective, ad; cons, kwargs...)
	u0, lb, ub = map(Base.Fix1(to_vec, vectorizer), (p, Corleone.get_bounds(prob.layer)...))
    return SciMLBase.OptimizationProblem(optf, u0, st; lb, ub, lcons=prob.lcons, ucons=prob.ucons, sense=sense)
end

"""
$(SIGNATURES)

Convenience constructor that first builds a [`CorleoneDynamicOptProblem`](@ref) from `layer`.
"""
function SciMLBase.OptimizationProblem(
    layer::Union{SingleShootingLayer,MultipleShootingLayer},
    ad::SciMLBase.ADTypes.AbstractADType, ;
    loss::Union{Symbol,Expr},
    constraints=[],
    kwargs...
)
    dynprob = CorleoneDynamicOptProblem(layer, loss, constraints...; kwargs...)
    return OptimizationProblem(dynprob, ad, vectorizer; kwargs...)
end
