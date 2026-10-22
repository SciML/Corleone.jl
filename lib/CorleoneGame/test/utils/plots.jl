# Time-series plots, constraint slacks, and display-label directives.
using CorleoneGame, Test

@testset "Visible floor and budget plots" begin
    g = Game(;
        name = "Plot regression", state_labels = ["Population"],
        control_labels = ["A", "B"], owners = [1, 2],
        dynamics! = (dx, x, u, t) -> fill!(dx, 0), objective = (x, i) -> x[2],
        x0 = [0.0, 0.0], T = 1.0, quadratures = [2], lower = zeros(2), upper = ones(2), initial = zeros(2),
        decode = x -> vcat(exp(x[1]), x[2:end]), constraints = [
            StateConstraint(; player = 1, index = 1, label = "population floor", lower = log(0.5), path = true),
            StateConstraint(; player = 2, index = 1, label = "population floor", lower = log(0.5), path = true),
            StateConstraint(; player = 1, index = 1, label = "terminal floor", lower = log(0.8)),
            StateConstraint(; player = 2, index = 2, label = "budget", upper = 1000.0),
        ]
    )
    t = [0.0, 0.5, 1.0]
    x = [0.0 log(0.9) log(0.8); 0.0 100.0 200.0]
    tr = (; t, x, physical = reduce(hcat, g.decode.(eachcol(x))))
    controls = zeros(2, 2)
    series = CorleoneGame.constraint_plot_series(g, tr)
    @test length(series) == 3
    @test occursin("A, B", series[1].label)
    @test series[1].slack ≈ log.([1.0, 0.9, 0.8] ./ 0.5)
    @test occursin("active", series[2].label)
    limits = CorleoneGame.iteration_plot_limits(g, [(tr, controls)])
    @test limits.states[2] < 2 # Exclude unplotted accumulators.
    @test limits.slack[2] > 1000
    @test limits.slack[1] < 0
    mktempdir() do dir
        for (name, ylimits) in (("timeseries", nothing), ("iteration", limits))
            path = joinpath(dir, "$name.png")
            f = CorleoneGame.timeseries(g, tr, controls, path; ylimits)
            @test filesize(path) > 1000
            axes = filter(a -> a isa CorleoneGame.Axis, f.content)
            @test length(axes) == 3
            @test axes[3].ylabel[] == "Constraint slack"
        end
    end
end

@testset "Plot directives and bounded legends" begin
    CG = CorleoneGame
    @test CG.plot_label("! hidden") === nothing
    @test CG.plot_label("SCALE 10 plant biomass") == (; label = "plant biomass", scale = 10.0)
    @test CG.plot_label("SCALE -2 signed value").scale == -2
    @test_throws ErrorException CG.plot_label("SCALE NaN broken")
    @test_throws ErrorException CG.plot_label("SCALE 10")
    labels = vcat(["! hidden", "SCALE 10 biomass"], ["state $i" for i in 3:20])
    entries = CG.plot_entries(labels)
    @test length(entries) == 16
    @test first(entries).index == 2
    @test last(entries).index == 17
    function fixture(; states = labels, inputs = labels, constraints = StateConstraint[])
        Game(;
            name = "Display rules", state_labels = states, control_labels = inputs, owners = ones(Int, length(inputs)),
            dynamics! = (dx, x, u, t) -> fill!(dx, 0), objective = (x, i) -> x[1],
            x0 = ones(length(states)), T = 1.0, quadratures = Int[], lower = zeros(length(inputs)),
            upper = ones(length(inputs)), initial = zeros(length(inputs)), constraints
        )
    end
    t = [0.0, 0.5, 1.0]; x = repeat(reshape(collect(1.0:20.0), :, 1), 1, 3)
    x[1, :] .= 1.0e6; x[18:end, :] .= 1.0e8
    tr = (; t, x, physical = x); u = ones(20, 2)
    g = fixture()
    lim = CG.iteration_plot_limits(g, [(tr, u)])
    @test 20 < lim.states[2] < 25 # Visible scaled biomass; hidden/truncated outliers excluded.
    @test 10 < lim.controls[2] < 11
    cs = [
        StateConstraint(; player = 1, index = 1, label = "! hidden floor", lower = 0.0, path = true),
        StateConstraint(; player = 1, index = 2, label = "SCALE 10 floor", lower = 1.0, path = true),
    ]
    append!(cs, [StateConstraint(; player = 1, index = i, label = "floor $i", lower = 0.0, path = true) for i in 3:20])
    constrained = fixture(; constraints = cs)
    series = CG.constraint_plot_series(constrained, tr)
    @test length(series) == 16
    @test first(series).slack == fill(10.0, 3)
    @test !occursin("SCALE", first(series).label)
    @test occursin("inactive", first(series).label)
    mktempdir() do dir
        for (name, game, expected, banks) in (
                ("empty", g, 2, 2), ("constraints", constrained, 3, 2),
                ("eight", fixture(; states = labels[3:10], inputs = labels[3:10]), 2, 1),
                ("hidden", fixture(; constraints = cs[1:1]), 2, 2),
            )
            f = CG.timeseries(game, tr, u, joinpath(dir, name * ".png"))
            axes = filter(a -> a isa CG.Axis, f.content)
            legends = filter(a -> a isa CG.Legend, f.content)
            @test length(axes) == expected
            @test all(a -> a.height[] == first(axes).height[], axes)
            @test all(l -> l.nbanks[] == banks, legends)
            @test length(first(axes).scene.plots) == (name == "eight" ? 8 : 16)
            if haskey(ENV, "CORLEONE_PLOT_PREVIEW")
                cp(joinpath(dir, name * ".png"), joinpath(ENV["CORLEONE_PLOT_PREVIEW"], name * ".png"); force = true)
            end
        end
    end
end
