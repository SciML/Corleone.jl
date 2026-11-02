"""
$(SIGNATURES)

Plot states, controls, and quadratures using the trajectory's symbolic metadata.
Controls are parameter histories, not appended state indices. The legacy
three-argument form accepts dimension counts but derives selections from metadata.
"""
plot_oc_problem(trajectory) = trajectoryplot(trajectory)
plot_oc_problem(trajectory, num_states, num_controls) = plot_oc_problem(trajectory)
