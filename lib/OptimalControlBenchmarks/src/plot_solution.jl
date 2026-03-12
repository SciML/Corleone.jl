function plot_oc_problem(sol, num_states, num_controls)
    f = Figure()
    ax = CairoMakie.Axis(f[1, 1])
    scatterlines!(ax, sol, vars = collect(1:1:num_states))
    f[1, 2] = Legend(f, ax, "States", framevisible = false)
    ax1 = CairoMakie.Axis(f[2, 1])
    stairs!(ax1, sol, vars = collect(num_states + 1:1:num_states + num_controls))
    f[2, 2] = Legend(f, ax1, "Controls", framevisible = false)
    return f
end
