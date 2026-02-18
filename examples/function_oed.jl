using Pkg
Pkg.activate(@__DIR__)
using Corleone
using CorleoneOED
using OrdinaryDiffEq
using SciMLSensitivity
using ComponentArrays
using LuxCore
using Random

using CairoMakie
using Optimization
using OptimizationMOI
using Ipopt
#using blockSQP
using LinearAlgebra

using CorleoneOED: Symbolics


t0, tf = -1.0, 1.0

x0 = [1.5]
p0 = [15.0, 2/5, 50.0]

fun(x,p) = [x[1]/5 + p[2]*x[1]^2 + 0.3*x[1]*sin(p[1]*x[1]) + 0.05*cos(p[3]*x[1]) + 0.1]

x = Symbolics.variables(:x, 1)
p = Symbolics.variables(:p, 1:length(p0))

eq = fun(x,p)

fp = Symbolics.jacobian(eq, p)

F = fp' * fp

J = Symbolics.build_function(fp, x, p; expression=Val{false}, cse=true)[1]
J = Base.Fix2(J, p0)


F_eq = Symbolics.build_function(F, x, p; expression=Val{false}, cse=true)[1]

function psi(x,p)
    F = weighted_F(p.w, p.x)
    A = J(x)' * J(x)
    return tr(F \ A)
end

function weighted_F(p, x)
    Jeval = J.(x)
    return sum(p .* [Ji' * Ji for Ji in Jeval])
end

function logdetF(p, x)
    F = weighted_F(p, x)
    return logdet(Symmetric(F))
end

### D-OPTIMAL

support_points = [[rand(t0:0.01:tf)] for _ = 1:6]
wi = inv(length(support_points)) * ones(length(support_points))

cons(res,x,p) = res .= sum(x)

optfun1 = OptimizationFunction(logdetF, AutoForwardDiff(), cons=cons)
optprob1 = OptimizationProblem(optfun1, wi, support_points, lb=zero(wi), ub=ones(length(wi)),
             sense=Optimization.MaxSense, lcons=[1.0], ucons=[1.0])
sol1 = solve(optprob1, Ipopt.Optimizer(), print_level=0)
wi = sol1.u

optfun = OptimizationFunction(psi, AutoForwardDiff())

optprob = OptimizationProblem(optfun, x0, (w=wi, x=support_points),
                lb=[t0], ub=[tf], sense=Optimization.MaxSense)


converged = false
@elapsed begin
    while !converged

        # Calculate new support point
        # Here, globally optimal by multistart (overkill in 1D)
        best_obj = 0.0
        new_support = []

        xtest = [rand(t0:1e-6:tf, 1) for _ = 1:2000]
        psieval = Base.Fix2(psi, (w=wi, x=support_points))
        best_idx = argmax(psieval.(xtest))
        new_support = xtest[best_idx]
        best_obj = psieval(new_support)

        #for x0 in xtest
        #    _sol = solve(remake(optprob, u0=x0, p=(w=wi, x=support_points)), Ipopt.Optimizer(), print_level = 0)
        #    if _sol.objective > best_obj
        #        best_obj = _sol.objective
        #        new_support = _sol.u
        #    end
        #end

        # Termination criteria: Objective <= #parameters
        # Else: Add found support point
        if best_obj <= length(p0) + 1e-6
            converged = true
            print("Found D-optimal design with $(length(support_points)) supports")
        else
            append!(wi, 1e-3)
            wi ./= sum(wi)
            push!(support_points, new_support)
        end


        # Update support weights wi
        sol1 = solve(remake(optprob1, u0=wi, p=support_points, lb=zero(wi),
                        ub=ones(length(wi))), Ipopt.Optimizer(), print_level = 0)
        wi = sol1.u

        # Prune unimportant support points
        idxs_to_keep = wi .> 1e-8
        wi = wi[idxs_to_keep]
        support_points = support_points[idxs_to_keep]
    end
end

converged

idxs = wi .> 1e-7
support_points_D = support_points[idxs]
wi_D = wi[idxs]

f = Figure()
ax = CairoMakie.Axis(f[1,1])
xtest = [x for x in t0:0.01:tf]
ytest = reduce(vcat, Base.Fix2(fun, p0).(xtest))
ysupp = reduce(vcat, Base.Fix2(fun, p0).(support_points_D))
lines!(ax, reduce(vcat, xtest), ytest)
scatter!(ax, reduce(vcat, support_points_D), ysupp, color=:black, marker=:x)
f


### A-optimal


support_points = [[rand(t0:0.01:tf)] for _ = 1:10]

wi = inv(length(support_points)) * ones(length(support_points))

function Aopt(p, x)
    F = Symmetric(weighted_F(p, x))
    return tr(inv(F))
end

function psiA(x, p)
    F = weighted_F(p.w, p.x)
    A = J(x)' * J(x)
    return tr(F \ A / F)   # F\A/F = F^{-1} * A * F^{-1}
end


weighted_F(wi, support_points)
Aopt(wi, support_points)

cons(res,x,p) = res .= sum(x)

optfun1 = OptimizationFunction(Aopt, AutoForwardDiff(), cons=cons)
optprob1 = OptimizationProblem(optfun1, wi, support_points, lb=zero(wi), ub=ones(length(wi)),
             sense=Optimization.MinSense, lcons=[1.0], ucons=[1.0])
sol1 = solve(optprob1, Ipopt.Optimizer(), print_level=0)
wi = sol1.u

optfun = OptimizationFunction(psiA, AutoForwardDiff())

converged = false
while !converged
    optprob = OptimizationProblem(optfun, x0, (w=wi, x=support_points),
                lb=[t0], ub=[tf], sense=Optimization.MaxSense)

    optprob.f(optprob.u0, optprob.p)

    xtest = [rand(-2:0.001:2.0, 1) for _ = 1:200]

    best_obj = 0.0
    new_support = []
    # Multistartoptimization for global solution, overkill for 1D
    for x0 in xtest
        _sol = solve(remake(optprob, u0=x0), Ipopt.Optimizer(), print_level=0)
        if _sol.objective > best_obj
            best_obj = _sol.objective
            new_support = _sol.u
        end
    end

    best_obj
    new_support

    if best_obj <= Aopt(wi, support_points) + 1e-4
        converged = true
        print("OPTIMAL")
    else
        append!(wi, 1e-3)
        wi ./= sum(wi)
        push!(support_points, new_support)
    end

    wi
    support_points

    sol1 = solve(remake(optprob1, u0=wi, p=support_points, lb=zero(wi),
                    ub=ones(length(wi))), Ipopt.Optimizer(), print_level=0)
    wi = sol1.u

    idxs_to_keep = wi .> 1e-8
    wi = wi[idxs_to_keep]
    support_points = support_points[idxs_to_keep]
end

converged

idxs = wi .> 0
support_points = support_points[idxs]
wi = wi[idxs]

support_points_A = copy(support_points)


f = Figure()
ax = CairoMakie.Axis(f[1,1])
xtest = [x for x in t0:0.01:tf]
ytest = reduce(vcat, Base.Fix2(fun, p0).(xtest))
ysupp = reduce(vcat, Base.Fix2(fun, p0).(support_points_A))
lines!(ax, reduce(vcat, xtest), ytest)
scatter!(ax, reduce(vcat, support_points_A), ysupp, color=:black, marker=:x)
f
