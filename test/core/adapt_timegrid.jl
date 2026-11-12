using Test, Corleone, LuxCore, OrdinaryDiffEqTsit5, ComponentArrays, Random

# Natural control value active at time t, read through the parameter layer.
control_value(layer, ps, st, t, i) = first(layer.parameter(t, ps.parameter, st.parameter))[i]

@testset "Refining and coarsening control grids" begin
    problem = ODEProblem((u, p, t) -> [p[2] - p[1] * u[1]], [1.0], (0.0, 4.0), [0.0, 0.5]; abstol = 1.0e-10, reltol = 1.0e-10)
    layer = ShootingLayer(problem, Tsit5(), Tunable(1; default = [1.0, 2.0, 3.0, 4.0], lowerbound = 0.0, upperbound = 4.2) => [1.0, 2.0, 3.0])
    ps, st = LuxCore.setup(MersenneTwister(1), layer)
    before = deepcopy((ps, st))
    fine = collect(0.5:0.5:3.5)
    new_ps, new_st = adapt_timegrid!(layer, ps, st, :p₁ => fine)
    @test (ps, st) == before # Inputs are not modified.
    @test new_ps.parameter.parameters.p₁.parameter.ps == [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]
    @test new_st.parameter.parameters.p₁.lookup.collection == fine
    @test new_st.timestops == Corleone.get_timestops(layer, nothing, new_st)
    for t in 0.0:0.25:4.0 # A refinement reproduces the control exactly.
        @test control_value(layer, new_ps, new_st, t, 1) == control_value(layer, ps, st, t, 1)
    end
    traj, _ = layer(nothing, new_ps, new_st)
    reference, _ = layer(nothing, ps, st)
    @test last(traj.u) ≈ last(reference.u)
    # Coarsening samples each new interval at its midpoint; a midpoint on a
    # switching time takes the following interval (right-continuous controls).
    coarse_ps, coarse_st = adapt_timegrid!(layer, ps, st, 1 => [2.0])
    @test coarse_ps.parameter.parameters.p₁.parameter.ps == [2.0, 4.0]
    @test adapt_timegrid!(layer, ps, st, 1 => [1.5])[1].parameter.parameters.p₁.parameter.ps == [1.0, 3.0]
    # Linear interpolation between interval midpoints, with constant or linear ends.
    linear_ps, _ = adapt_timegrid!(layer, ps, st, :p₁ => fine; interpolation = :linear)
    @test linear_ps.parameter.parameters.p₁.parameter.ps ≈ [1.0, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.0]
    extrapolated, _ = adapt_timegrid!(layer, ps, st, :p₁ => fine; interpolation = :linear, extrapolation = :linear)
    values = extrapolated.parameter.parameters.p₁.parameter.ps
    @test first(values) ≈ 0.75 # Extends the first slope.
    @test last(values) ≈ 4.2 # 4.25 is clipped to the upper bound.
    clipped, _ = adapt_timegrid!(layer, ps, st, :p₁ => [0.1, 3.9]; interpolation = :linear, extrapolation = :linear)
    @test clipped.parameter.parameters.p₁.parameter.ps ≈ [0.55, 2.5, 4.2] # Bounds clip extrapolation.
    # A constant parameter can become a control, and a control a constant.
    untimed_ps, untimed_st = adapt_timegrid!(layer, ps, st, :p₁ => nothing)
    @test untimed_ps.parameter.parameters.p₁.parameter.ps == [3.0] # Value at the span midpoint t = 2.
    @test isnothing(untimed_st.parameter.parameters.p₁.lookup.collection)
    # ComponentArray parameters keep their structure.
    component_ps, _ = adapt_timegrid!(layer, ComponentArray(ps), st, :p₁ => fine)
    @test component_ps isa ComponentArray
    @test component_ps.parameter.parameters.p₁.parameter.ps == new_ps.parameter.parameters.p₁.parameter.ps
    # Invalid requests.
    @test_throws ArgumentError adapt_timegrid!(layer, ps, st, :p₁ => [2.0, 1.0])
    @test_throws DomainError adapt_timegrid!(layer, ps, st, :p₁ => [0.0, 1.0])
    @test_throws ArgumentError adapt_timegrid!(layer, ps, st, :u₁ => [1.0])
    @test_throws ArgumentError adapt_timegrid!(layer, ps, st, :unknown => [1.0])
    @test_throws ArgumentError adapt_timegrid!(layer, ps, st, :p₁ => [1.0]; interpolation = :cubic)
    # p₂ is untimed and tunable; it shares no grid with p₁, so it can be adapted alone.
    p2_ps, p2_st = adapt_timegrid!(layer, ps, st, :p₂ => [2.0])
    @test p2_ps.parameter.parameters.p₂.parameter.ps == [0.5, 0.5]
    @test p2_st.parameter.parameters.p₂.lookup.collection == [2.0]
    @test control_value(layer, p2_ps, p2_st, 3.0, 2) == 0.5
end

@testset "Shared grids, fixed members, integers and shooting segments" begin
    problem = ODEProblem((u, p, t) -> [p[1] - u[1] + p[2] * p[3]], [1.0], (0.0, 2.0), [0.3, 0.2, 2.0]; abstol = 1.0e-11, reltol = 1.0e-11)
    layer = ShootingLayer(
        problem, Tsit5(),
        Tunable(1; default = [0.0, 1.0], lowerbound = 0.0, upperbound = 3.0, integer = true) => [1.0],
        Tunable(2; default = 0.2) => nothing,
        Constant(3; default = 2.0) => nothing, # Shares the untimed grid with p₂.
        Tunable(1; kind = Val(:variable), default = 1.0) => [1.0]
    )
    ps, st = LuxCore.setup(MersenneTwister(1), layer)
    @test_throws ArgumentError adapt_timegrid!(layer, ps, st, :p₂ => [0.5]) # p₃ shares the grid.
    @test_throws ArgumentError adapt_timegrid!(layer, ps, st, :p₂ => [0.5], :p₃ => [1.5])
    shared_ps, shared_st = adapt_timegrid!(layer, ps, st, :p₂ => [0.5], :p₃ => [0.5])
    group = shared_st.parameter.parameters.p₂
    @test group.lookup.collection == [0.5]
    @test shared_ps.parameter.parameters.p₂.parameter.p₂.ps == [0.2, 0.2]
    @test group.parameter.p₃.parameters.ps == [2.0, 2.0] # Fixed members are resampled in state.
    # Integer controls are rounded after linear interpolation.
    rounded, _ = adapt_timegrid!(layer, ps, st, :p₁ => [0.5, 1.0, 1.5]; interpolation = :linear)
    @test rounded.parameter.parameters.p₁.parameter.ps == [0.0, 0.0, 1.0, 1.0]
    # Matching equalities are reinserted at shooting nodes on the new grid.
    segmented_ps, segmented_st = add_parameter_segments!(layer, ps, st)
    model_count = get_number_of_shooting_constraints(layer, segmented_ps, segmented_st)
    adapted_ps, adapted_st = adapt_timegrid!(layer, segmented_ps, segmented_st, :p₂ => [0.25, 1.5], :p₃ => [0.25, 1.5])
    @test 1.0 in adapted_st.parameter.parameters.p₂.lookup.collection
    @test !isempty(adapted_st.parameter.parameters.p₂.lookup.inserted)
    @test get_number_of_shooting_constraints(layer, adapted_ps, adapted_st) == model_count
    traj, _ = layer(nothing, adapted_ps, adapted_st)
    reference, _ = layer(nothing, segmented_ps, segmented_st)
    @test last(traj.u) ≈ last(reference.u)
end
