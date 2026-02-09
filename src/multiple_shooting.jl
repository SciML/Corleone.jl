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

get_quadrature_indices(layer::MultipleShootingLayer) = get_quadrature_indices(layer.layer)

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

function _parallel_solve(
        shooting::MultipleShootingLayer,
        u0,
        ps,
        st::NamedTuple{fields},
    ) where {fields}
    args = collect(
        ntuple(
            i -> (u0, __getidx(ps, fields[i]), __getidx(st, fields[i]), i > 1), length(st)
        )
    )
    return mythreadmap(shooting.ensemble_alg, Base.Splat(shooting.layer), args)
end

function (shooting::MultipleShootingLayer)(u0, ps, st::NamedTuple{fields}) where {fields}
    ret = Corleone._parallel_solve(shooting, u0, ps, st)
    u = first.(ret)
    sts = NamedTuple{fields}(last.(ret))
    return Trajectory(u, sts, get_quadrature_indices(shooting)), sts
end

function Trajectory(u::AbstractVector{TR}, sts, quadrature_indices) where {TR <: Trajectory}
    size(u, 1) == 1 && return only(u)
    p = first(u).p
    sys = first(u).sys
    us = map(state_values, u)
    ts = map(current_time, u)
    tnew = reduce(
        vcat, map(i -> i == lastindex(ts) ? ts[i] : ts[i][1:(end - 1)], eachindex(ts))
    )
    offsets = cumsum(map(i -> lastindex(us[i]), eachindex(us[1:(end - 1)])))
    shooting_val_1 = ((u0 = eltype(first(first(us)))[], p = eltype(p)[], controls = eltype(first(first(us)))[]))
    shooting_vals = map(eachindex(us[1:(end - 1)])) do i
        uprev = us[i]
        unext = us[i + 1]
        idx = sts[i + 1].shooting_indices
        nx = statelength(sts[i + 1].initial_condition)
        controlidx = setdiff(idx, Base.OneTo(nx))
        stateidx = setdiff(idx, controlidx)

        (
            u0 = last(uprev)[stateidx] .- first(unext)[stateidx],
            p = u[i].p .- u[i + 1].p,
            controls = last(uprev)[controlidx] .- first(unext)[controlidx],
        )
    end
    shootings = NamedTuple{(keys(sts)...,)}(
        (
            shooting_val_1,
            shooting_vals...,
        )
    )
    # Sum up the quadratures
    q_prev = us[1][end][quadrature_indices]
    for i in eachindex(us)[2:end]
        for j in eachindex(us[i])
            us[i][j][quadrature_indices] += q_prev
        end
        q_prev = us[i][end][quadrature_indices]
    end
    unew = reduce(
        vcat, map(i -> i == lastindex(us) ? us[i] : us[i][1:(end - 1)], eachindex(us))
    )
    return Trajectory(sys, unew, p, tnew, shootings, offsets)
end

function get_number_of_state_matchings(
        shooting::MultipleShootingLayer,
        ps = LuxCore.initialparameters(Random.default_rng(), shooting),
        st = LuxCore.initialstates(Random.default_rng(), shooting),
    )
    return sum(xi -> size(intersect(xi.shooting_indices, Base.OneTo(statelength(xi.initial_condition))), 1), Base.tail(st))
end

function get_number_of_parameter_matchings(
        shooting::MultipleShootingLayer,
        ps = LuxCore.initialparameters(Random.default_rng(), shooting),
        st = LuxCore.initialstates(Random.default_rng(), shooting),
    )
    return sum(xi -> size(xi.p, 1), Base.front(ps))
end

function get_number_of_control_matchings(
        shooting::MultipleShootingLayer,
        ps = LuxCore.initialparameters(Random.default_rng(), shooting),
        st = LuxCore.initialstates(Random.default_rng(), shooting),
    )
    return sum(xi -> size(setdiff(xi.shooting_indices, Base.OneTo(statelength(xi.initial_condition))), 1), Base.tail(st))
end

function get_number_of_shooting_constraints(
        shooting::MultipleShootingLayer,
        ps = LuxCore.initialparameters(Random.default_rng(), shooting),
        st = LuxCore.initialstates(Random.default_rng(), shooting),
    )
    return get_number_of_state_matchings(shooting, ps, st) +
        get_number_of_control_matchings(shooting, ps, st) +
        get_number_of_parameter_matchings(shooting, ps, st)
end

deepvcat(V::AbstractVector) = V
deepvcat(NTV::NamedTuple) = reduce(vcat, NTV |> values .|> deepvcat)

"""
    stage_ordered_shooting_constraints(traj)
    
Returns the shooting violations sorted by shooting-stage
and per-stage sorted by states - parameters - controls
"""
stage_ordered_shooting_constraints(traj::Trajectory) = deepvcat(traj.shooting)

function collect_into!(res::AbstractVector, sval::SV, ind::Vector{Int64} = [0]) where {SV <: AbstractVector}
    for i in eachindex(sval)
        res[ind[1] += 1] = sval[i]
    end
    return
end
function collect_into!(res::AbstractVector, sval::NamedTuple, ind::Vector{Int64} = [0])
    for key in keys(sval)
        collect_into!(res, sval[key], ind)
    end
    return
end

"""
    stage_ordered_shooting_constraints!(res, traj)
    
In-place version of `stage_ordered_shooting_constraints`\" function`
"""
function stage_ordered_shooting_constraints!(res::AbstractVector, traj::Trajectory{S, U, P, T, SH}) where {S, U, P, T, SH <: NamedTuple}
    collect_into!(res, traj.shooting)
    return res
end


function _matchings(traj::Trajectory{S, U, P, T, SH}, kind::Symbol) where {S, U, P, T, SH <: NamedTuple}
    return map(keys(traj.shooting)) do key
        traj.shooting[key][kind]
    end |> Base.Fix1(reduce, vcat)
end
state_matchings(traj::Trajectory{S, U, P, T, SH}) where {S, U, P, T, SH <: NamedTuple} = _matchings(traj, :u0)
parameter_matchings(traj::Trajectory{S, U, P, T, SH}) where {S, U, P, T, SH <: NamedTuple} = _matchings(traj, :p)
control_matchings(traj::Trajectory{S, U, P, T, SH}) where {S, U, P, T, SH <: NamedTuple} = _matchings(traj, :controls)

function _matchings!(res::AbstractVector, traj::Trajectory{S, U, P, T, SH}, kind::Symbol, ind::Vector{Int64} = [1]) where {S, U, P, T, SH <: NamedTuple}
    for key in keys(traj.shooting)
        res[UnitRange(ind[1], (ind[1] += length(traj.shooting[key][kind])) - 1)] = traj.shooting[key][kind]
    end
    return res
end

state_matchings!(res::AbstractVector, traj::Trajectory{S, U, P, T, SH}) where {S, U, P, T, SH <: NamedTuple} = _matchings!(res, traj, :u0)
parameter_matchings!(res::AbstractVector, traj::Trajectory{S, U, P, T, SH}) where {S, U, P, T, SH <: NamedTuple} = _matchings!(res, traj, :p)
control_matchings!(res::AbstractVector, traj::Trajectory{S, U, P, T, SH}) where {S, U, P, T, SH <: NamedTuple} = _matchings!(res, traj, :controls)

"""
    shooting_constraints(traj)
    
Returns the shooting violations sorted sorted by states - parameters - controls
and per-kind sorted by shooting-stage.
"""
shooting_constraints(traj::Trajectory{S, U, P, T, SH}) where {S, U, P, T, SH <: NamedTuple} = vcat((_matchings(traj, kind) for kind in (:u0, :p, :controls))...)

"""
    shooting_constraints!(res, traj)
    
In-place version of `shooting_constraints`.
"""
function shooting_constraints!(res::AbstractVector, traj::Trajectory{S, U, P, T, SH}) where {S, U, P, T, SH <: NamedTuple}
    ind = [1]
    for kind in (:u0, :p, :controls)
        _matchings!(res, traj, kind, ind)
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
