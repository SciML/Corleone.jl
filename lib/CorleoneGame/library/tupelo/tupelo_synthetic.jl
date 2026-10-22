# Generate noisy observations from output/trajectory.csv.
# Run from lib/CorleoneGame with julia --project=. library/tupelo/tupelo_synthetic.jl.
using Random
using Printf

const T = 8.0
const measurement_times = [
    0.0, 0.17, 0.63, 1.08, 1.57, 2.04, 2.71, 3.02, 3.66, 4.11,
    4.58, 5.03, 5.47, 5.92, 6.21, 6.83, 7.14, 7.56, 7.82, 8.0,
]
const sigma = 0.02
const seed = 20260908
const input_file = joinpath(@__DIR__, "output", "trajectory.csv")
const output_file = joinpath(@__DIR__, "output", "tupelo_synthetic.csv")

function read_trajectory(path)
    lines = readlines(path)
    rows = [parse.(Float64, split(line, ',')) for line in lines[2:end] if !isempty(line)]
    return (t = [r[1] for r in rows], x = reduce(hcat, [[r[2], r[3], r[4]] for r in rows]))
end

function interpolate(t, x, tq)
    tq <= t[1] && return x[:, 1]
    tq >= t[end] && return x[:, end]
    k = searchsortedlast(t, tq)
    α = (tq - t[k]) / (t[k + 1] - t[k])
    return (1 - α) .* x[:, k] .+ α .* x[:, k + 1]
end

function main()
    isfile(input_file) || error("Missing $input_file; run tupelo.jl first")
    trajectory = read_trajectory(input_file)
    rng = MersenneTwister(seed)
    open(output_file, "w") do io
        println(io, "time,x_1,x_2,x_3,sigma,seed")
        for time in measurement_times
            truth = interpolate(trajectory.t, trajectory.x, time)
            observation = truth .+ sigma .* randn(rng, 3)
            observed = [@sprintf("%.12g", y) for y in observation]
            println(
                io, join(
                    vcat(
                        [@sprintf("%.12g", time)], observed,
                        [string(sigma), string(seed)]
                    ), ','
                )
            )
        end
    end
    return println("Wrote ", length(measurement_times), " measurements to ", output_file)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
