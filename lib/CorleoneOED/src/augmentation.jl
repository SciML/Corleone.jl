# Adapted from Christoph Plate (@chplate), AlCap23/Corleone.jl, commit 7443580115.
# See the attribution section in ../README.md.

function augment_system(
        prob::SciMLBase.AbstractDEProblem, alg::SciMLBase.AbstractDEAlgorithm;
        svd = false,
        kwargs...
    )
    sys = ControlSymbolCache(prob)
    states = collect(SymbolicIndexingInterface.variable_symbols(sys))
    sort!(states, by = Base.Fix1(SymbolicIndexingInterface.variable_index, sys))
    dstates = map(xi -> Symbol(:d, xi), states)
    ps = collect(SymbolicIndexingInterface.parameter_symbols(sys))
    sort!(ps, by = Base.Fix1(SymbolicIndexingInterface.parameter_index, sys))
    ts = SymbolicIndexingInterface.independent_variable_symbols(sys)
    vars = Symbolics.variable.(Symbol.(states))
    vars = Symbolics.setdefaultval.(vars, vec(prob.u0))
    parameters = Symbolics.variable.(Symbol.(ps))
    p0, _, _ = SciMLStructures.canonicalize(SciMLStructures.Tunable(), prob.p)
    parameters = Symbolics.setdefaultval.(parameters, p0)
    config = (;
        differential_vars = Symbolics.variable.(dstates),
        vars = vars,
        parameters = parameters,
        independent_vars = Symbolics.variable.(ts),
    )
    config = symbolify_equations(prob, config; kwargs...)
    config = add_observed_equations(config; kwargs...)
    if svd
        config = derive_sensitivity_equations(prob, config; basis = sensitivity_basis(prob, alg, config; kwargs...), kwargs...)
    else
        config = derive_sensitivity_equations(prob, config; kwargs...)
    end
    problem, observed = finalize_config(prob, config; kwargs...)
    return problem, merge(observed, (; discrete_ranges = Tuple(config.discrete_ranges))), config.basis
end

function symbolify_equations(prob::SciMLBase.AbstractDEProblem, config; kwargs...)
    (; vars, parameters, independent_vars) = config
    f = prob.f
    eqs = if SciMLBase.isinplace(prob)
        out = zero(vars)
        f(out, vars, parameters, only(independent_vars))
        out
    else
        f(vars, parameters, only(independent_vars))
    end
    return merge(config, (; equations = eqs))
end

function symbolify_equations(prob::SciMLBase.DAEProblem, config; kwargs...)
    (; differential_vars, vars, parameters, independent_vars) = config
    f = prob.f
    eqs = if SciMLBase.isinplace(prob)
        out = zero(vars)
        f(out, differential_vars, vars, parameters, only(independent_vars))
        out
    else
        f(differential_vars, vars, parameters, only(independent_vars))
    end
    return merge(config, (; equations = eqs))
end

function compute_initial_F(prob::Union{ODEProblem, DAEProblem}, alg, config, params)
    (; vars, parameters, independent_vars) = config
    observed_jacobian = vcat(config.observed_continuous_jacobian, config.observed_discrete_jacobian)

    sol = solve(prob, alg, abstol = 1.0e-10, reltol = 1.0e-8)
    pred(p) = begin
        prob_ = remake(prob, p = p)
        Array(solve(prob_, alg, saveat = sol.t))[:]
    end

    hx = Symbolics.build_function(observed_jacobian, vars, parameters, only(independent_vars); expression = Val{false}, cse = true)[1]
    observed = vcat(config.observed_continuous, config.observed_discrete)
    hp = Symbolics.build_function(Symbolics.jacobian(observed, parameters), vars, parameters, only(independent_vars); expression = Val{false}, cse = true)[1]
    G = ForwardDiff.jacobian(pred, prob.p)

    nx = size(prob.u0, 1)
    F_tf_sens = sum(
        map(enumerate(diff(sol.t))) do (i, Δt)
            G_t = G[(i * nx + 1):((i + 1) * nx), params]
            x = sol.u[i + 1]
            hx_ = hx(x, prob.p, sol.t[i + 1])
            J = hx_ * G_t + hp(x, prob.p, sol.t[i + 1])[:, params]
            Δt * (J' * J)
        end
    )

    return F_tf_sens
end

function compute_svd_of_F(prob, alg, config, params; ns = nothing, threshold_singular_values = 0.95, threshold_singular_vectors = 0.1, kwargs...)
    F = compute_initial_F(prob, alg, config, params)
    svdF = svd(F)

    0 <= threshold_singular_values < 1 || throw(ArgumentError("The singular-value threshold must be in [0, 1)."))
    0 <= threshold_singular_vectors < 1 || throw(ArgumentError("The singular-vector threshold must be in [0, 1)."))
    if isnothing(ns)
        iszero(sum(abs2, svdF.S)) && throw(ArgumentError("The nominal design has no reference information for SVD reduction."))
        ns = findfirst(>=(threshold_singular_values), cumsum(abs2.(svdF.S)) / sum(abs2, svdF.S))
    end
    ns isa Integer && 1 <= ns <= length(params) || throw(ArgumentError("ns must be between 1 and the number of estimated parameters."))

    important_params = begin
        U = svdF.U[:, 1:ns]
        U_important = abs.(U) .> threshold_singular_vectors
        map(x -> any(x), eachrow(U_important))
    end
    return svdF, ns, important_params
end

function sensitivity_basis(prob, alg, config; params, kwargs...)
    svdF, ns, important = compute_svd_of_F(prob, alg, config, params; kwargs...)
    basis = zeros(eltype(svdF.U), length(params), ns)
    basis[important, :] = svdF.U[important, 1:ns]
    return basis
end

function derive_sensitivity_equations(
        prob, config; params,
        basis = Matrix{eltype(prob.u0)}(I, length(params), length(params)), kwargs...
    )
    (; differential_vars, vars, parameters, equations) = config
    nx, ns = length(vars), size(basis, 2)
    dG = Symbolics.variables(:dG, 1:nx, 1:ns)
    G = Symbolics.setdefaultval.(Symbolics.variables(:G, 1:nx, 1:ns), zeros(eltype(prob.u0), nx, ns))
    dfdx = Symbolics.jacobian(equations, vars)
    dfdp = Symbolics.jacobian(equations, parameters[params])
    equations_G = dfdx * G + dfdp * basis
    if prob isa SciMLBase.DAEProblem
        equations_G += Symbolics.jacobian(equations, differential_vars) * dG
    end
    return merge(
        config, (;
            sensitivities = G, differential_sensitivities = dG,
            sensitivity_equations = equations_G, basis,
        )
    )
end

# Trace each observation once and retain the row block sharing its weight.
function measurement_expressions(measurements, vars, parameters, time)
    outputs = map(measurements) do measurement
        observation_values(measurement.observed(vars, parameters, time))
    end
    lengths = map(length, outputs)
    ranges = map(cumsum(lengths), lengths) do stop, n
        (stop - n + 1):stop
    end
    return reduce(vcat, outputs; init = Num[]), ranges
end

function add_observed_equations(
        config; continuous_measurements = ContinuousMeasurement[],
        discrete_measurements = DiscreteMeasurement[], kwargs...
    )
    (; vars, parameters, independent_vars) = config
    time = only(independent_vars)
    continuous, continuous_ranges = measurement_expressions(continuous_measurements, vars, parameters, time)
    discrete, discrete_ranges = measurement_expressions(discrete_measurements, vars, parameters, time)
    return merge(
        config, (;
            observed_continuous = continuous,
            observed_continuous_jacobian = Symbolics.jacobian(continuous, vars),
            continuous_ranges,
            observed_discrete = discrete,
            observed_discrete_jacobian = Symbolics.jacobian(discrete, vars),
            discrete_ranges,
        )
    )
end

function finalize_config(
        prob, config; params, continuous_measurements = ContinuousMeasurement[],
        discrete_measurements = DiscreteMeasurement[], kwargs...
    )
    (; differential_vars, vars, parameters, equations) = config
    (; sensitivities, differential_sensitivities, sensitivity_equations) = config
    (; observed_continuous_jacobian, observed_continuous) = config
    (; observed_discrete_jacobian, observed_discrete) = config

    n = size(sensitivities, 2)

    selector = triu(trues(n, n))
    F = Symbolics.variables(:F, 1:n, 1:n)
    F = Symbolics.setdefaultval.(F, zero(eltype(prob.u0)))
    dF = Symbolics.variables(:dF, 1:n, 1:n)
    J_cont = observed_continuous_jacobian * sensitivities + Symbolics.jacobian(observed_continuous, parameters[params]) * config.basis
    J_disc = observed_discrete_jacobian * sensitivities + Symbolics.jacobian(observed_discrete, parameters[params]) * config.basis

    disc_names = [x.id for x in discrete_measurements]
    cont_names = [x.id for x in continuous_measurements]

    w_disc = Symbolics.variable.(disc_names)
    w_disc = Symbolics.setdefaultval.(w_disc, one(eltype(prob.u0)))

    w_cont = Symbolics.variable.(cont_names)
    w_cont = Symbolics.setdefaultval.(w_cont, one(eltype(prob.u0)))

    F_cont = !isempty(w_cont) ? sum(enumerate(w_cont)) do (i, wi)
            rows = config.continuous_ranges[i]
            J = J_cont[rows, :]
            wi * J'J
    end : zero.(F)
    append!(parameters, w_disc)
    append!(parameters, w_cont)

    F_eqs = vec(F_cont[selector])
    fisher = [selector[i, j] ? F[i, j] : F[j, i] for i in 1:n, j in 1:n]
    F = F[selector]
    dF = dF[selector]
    if isa(prob, DAEProblem)
        F_eqs = vec(dF) .- F_eqs
    end
    new_vars = vcat(vars, vec(sensitivities), vec(F))
    new_differential_vars = vcat(differential_vars, vec(differential_sensitivities), vec(dF))
    new_equations = vcat(equations, vec(sensitivity_equations), vec(F_eqs))
    config = merge(
        config, (;
            vars = new_vars, differential_vars = new_differential_vars, equations = new_equations,
            observed = (;
                fisher = fisher,
                sensitivities = sensitivities,
                hx_G_discrete = J_disc,
                local_information_gain = Num.(vcat(J_cont, J_disc)),
                observed_continuous = observed_continuous,
                observed_discrete = observed_discrete,
            ),
        )
    )
    return build_new_system(prob, config)
end

function build_new_system(prob, config)
    (; vars, parameters, independent_vars, observed) = config
    u0 = Symbolics.getdefaultval.(vars)
    p0 = Symbolics.getdefaultval.(parameters)
    defaults = Dict(vcat(Symbol.(vars), Symbol.(parameters)) .=> vcat(u0, p0))
    sys = SymbolCache(
        Symbol.(vars), Symbol.(parameters), only(Symbol.(independent_vars)); defaults
    )
    problem = build_augmented_problem(prob, config, sys, u0, p0)
    getters = map(observed) do expression
        f = first(
            Symbolics.build_function(
                expression, vars, parameters, only(independent_vars);
                expression = Val{false}, cse = true
            )
        )
        ObservationGetter(f)
    end
    return problem, getters
end

function build_augmented_problem(prob::ODEProblem, config, sys, u0, p0)
    (; equations, vars, parameters, independent_vars, sensitivities) = config
    inplace = SciMLBase.isinplace(prob)
    foop, fiip = Symbolics.build_function(equations, vars, parameters, only(independent_vars); expression = Val{false}, cse = true)
    ns = size(sensitivities, 2)
    nf = length(u0) - length(prob.u0) * (ns + 1)
    mass_matrix = augmented_mass_matrix(prob, ns, nf)
    f = ODEFunction{inplace, SciMLBase.FullSpecialize}(inplace ? fiip : foop; sys, mass_matrix)
    return remake(prob; f, u0, p = p0)
end

function build_augmented_problem(prob::DAEProblem, config, sys, u0, p0)
    (; equations, vars, differential_vars, parameters, independent_vars, sensitivities) = config
    inplace = SciMLBase.isinplace(prob)
    foop, fiip = Symbolics.build_function(equations, differential_vars, vars, parameters, only(independent_vars); expression = Val{false}, cse = true)
    du0 = vcat(prob.du0, zeros(eltype(u0), length(u0) - length(prob.du0)))
    residual = foop(du0, u0, p0, first(prob.tspan))
    jacobian = ForwardDiff.jacobian(du -> foop(du, u0, p0, first(prob.tspan)), du0)
    du0 -= pinv(jacobian) * residual

    ns = size(sensitivities, 2)
    nf = length(u0) - length(prob.u0) * (ns + 1)
    flags = isnothing(prob.differential_vars) ? nothing : vcat(
            prob.differential_vars, repeat(prob.differential_vars, ns), trues(nf),
        )
    f = DAEFunction(inplace ? fiip : foop; sys)
    return remake(prob; f, du0, u0, p = p0, differential_vars = flags)
end

function augmented_mass_matrix(prob, ns, nf)
    mass = prob.f.mass_matrix
    mass == I && return I
    mass isa Union{UniformScaling, AbstractMatrix} || throw(ArgumentError("OED augmentation requires a constant mass matrix."))
    nx = length(prob.u0)
    block = mass isa UniformScaling ? Matrix(mass, nx, nx) : Matrix(mass)
    return cat(block, ntuple(_ -> block, ns)..., Matrix{eltype(block)}(I, nf, nf); dims = (1, 2))
end

# `getsym` treats integer constants as indices. Numerical observation functions
# preserve constants and direct parameter dependence in all observation matrices.
struct ObservationGetter{F}
    f::F
end
function (getter::ObservationGetter)(trajectory::Trajectory)
    return map(trajectory.u, trajectory.t) do u, t
        p = SymbolicIndexingInterface.parameter_values_at_time(trajectory, trajectory, t)
        getter.f(u, p, t)
    end
end
