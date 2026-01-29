struct MultipleShootingLayer{L, I, E, Z} <: LuxCore.AbstractLuxWrapperLayer{:layer}
    "The original layer"
    layer::L
    "The shooting intervals"
    shooting_intervals::I
    "The ensemble algorithm"
    ensemble_alg::E
    "The initialization scheme"
    initialization::Z
end

function Base.show(io::IO, layer::MultipleShootingLayer)
    type_color, no_color = SciMLBase.get_colorizers(io)
    print(
        io,
        type_color,
        "MultipleShootingLayer ",
        no_color,
        "with $(length(layer.shooting_intervals)) shooting intervals and $(length(get_controls(layer.layer))) controls.\n",
    )
    print(io, "Underlying problem: ")
    return print(io, layer.layer)
end

"""
$(FUNCTIONNAME)

Initializes all shooting nodes with their default value, i.e., their initial value in
the underlying problem.
"""
function default_initialization(rng::Random.AbstractRNG, shooting::MultipleShootingLayer)
    (; shooting_intervals, layer) = shooting
    names = ntuple(i -> Symbol(:interval, "_", i), length(shooting_intervals))
    vals = ntuple(
        i -> __initialparameters(
            rng, layer; tspan = shooting_intervals[i], shooting_layer = i != 1
        ),
        length(shooting_intervals),
    )
    return NamedTuple{names}(vals)
end

function MultipleShootingLayer(prob, alg, tpoints::AbstractVector; kwargs...)
    return MultipleShootingLayer(prob, alg, tpoints...; kwargs...)
end

function MultipleShootingLayer(
        prob::SciMLBase.AbstractDEProblem,
        alg::SciMLBase.DEAlgorithm,
        tpoints::Real...;
        ensemble_alg = EnsembleSerial(),
        initialization = default_initialization,
        kwargs...,
    )
    layer = SingleShootingLayer(prob, alg; kwargs...)
    return MultipleShootingLayer(layer, tpoints...; ensemble_alg, initialization, kwargs...)
end

function MultipleShootingLayer(
        layer,
        tpoints::Real...;
        ensemble_alg = EnsembleSerial(),
        initialization = default_initialization,
        kwargs...,
    )
    tspans = vcat(collect(tpoints), collect(layer.problem.tspan))
    sort!(tspans)
    unique!(tspans)
    tspans = [tispan for tispan in zip(tspans[1:(end - 1)], tspans[2:end])]
    tspans = tuple(tspans...)
    return MultipleShootingLayer{
        typeof(layer), typeof(tspans), typeof(ensemble_alg), typeof(initialization),
    }(
        layer, tspans, ensemble_alg, initialization
    )
end

function LuxCore.initialparameters(rng::Random.AbstractRNG, shooting::MultipleShootingLayer)
    (; initialization) = shooting
    return initialization(rng, shooting)
end

function LuxCore.parameterlength(shooting::MultipleShootingLayer)
    return last(get_block_structure(shooting))
end

function LuxCore.initialstates(rng::Random.AbstractRNG, shooting::MultipleShootingLayer)
    (; shooting_intervals, layer) = shooting
    names = ntuple(i -> Symbol(:interval, "_", i), length(shooting_intervals))
    vals = ntuple(
        i ->
        __initialstates(rng, layer; tspan = shooting_intervals[i], shooting_layer = i != 1),
        length(shooting_intervals),
    )
    return NamedTuple{names}(vals)
end

function (shooting::MultipleShootingLayer)(u0, ps, st::NamedTuple{fields}) where {fields}
    (; layer, ensemble_alg) = shooting
    ret = Corleone._parallel_solve(ensemble_alg, layer, u0, ps, st)
    u = first.(ret)
    sts = NamedTuple{fields}(last.(ret))
    return Trajectory(u, sts), sts
end

function Trajectory(u::AbstractVector, sts)
    size(u, 1) == 1 && return only(u)
    p = first(u).p
    sys = first(u).sys
    us = map(state_values, u)
    ts = map(current_time, u)
    tnew = reduce(
        vcat, map(i -> i == lastindex(ts) ? ts[i] : ts[i][1:(end - 1)], eachindex(ts))
    )
    offsets = cumsum(map(i -> lastindex(us[i]), eachindex(us[1:(end - 1)])))
    shootings = map(eachindex(us[1:(end - 1)])) do i
        uprev = us[i]
        unext = us[i + 1]
        idx = sts[i + 1].shooting_indices
        vcat(last(uprev)[idx] .- first(unext)[idx], u[i].p .- u[i + 1].p)
    end
    unew = reduce(
        vcat, map(i -> i == lastindex(us) ? us[i] : us[i][1:(end - 1)], eachindex(us))
    )
    return Trajectory(sys, unew, p, tnew, shootings, offsets)
end

function get_number_of_shooting_constraints(
        shooting::MultipleShootingLayer,
        ps = LuxCore.initialparameters(Random.default_rng(), shooting),
        st = LuxCore.initialstates(Random.default_rng(), shooting),
    )
    return sum(xi -> size(xi.shooting_indices, 1), Base.tail(st)) +
        sum(xi -> size(xi.p, 1), Base.front(ps))
end

function shooting_constraints(traj::Trajectory)
    return reduce(vcat, shooting_violations(traj))
end

function shooting_constraints!(res::AbstractVector, traj::Trajectory)
    i = 0
    for subvec in traj.shooting, j in eachindex(subvec)
        i += 1
        res[i] = subvec[j]
    end
    return res
end

"""
    get_block_structure(layer)

Compute the block structure of the hessian of the Lagrangian of an optimal control problem
as specified via the `shooting_intervals` of the `MultipleShootingLayer`.
"""
function get_block_structure(mslayer::MultipleShootingLayer)
    (; layer, shooting_intervals) = mslayer
    ps_lengths = collect(
        map(enumerate(shooting_intervals)) do (i, tspan)
            __parameterlength(layer; tspan = tspan, shooting_layer = i > 1)
        end,
    )
    return vcat(0, cumsum(ps_lengths))
end

function get_bounds(mslayer::MultipleShootingLayer)
    (; layer, shooting_intervals) = mslayer
    names = ntuple(i -> Symbol(:interval, "_", i), length(shooting_intervals))
    bounds = map(enumerate(shooting_intervals)) do (i, tspan)
        get_bounds(layer; tspan = tspan, shooting = i > 1)
    end
    return NamedTuple{names}(first.(bounds)), NamedTuple{names}(last.(bounds))
end
