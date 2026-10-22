# Alternative tupelo instance for Section 5.3 (allowance variant).
# Compared with the reference instance (tupelo.jl, mapped from Low et al. 2013), it
#   restrictions: enables a carbon allowance (C_A) and effort allowances (U_A, U_B);
#   mechanism:    uses saturating tissue availability rho(L)=L/(K+L), egg mortality,
#                 mild defense lethality, and exponentially discounted leaf value;
#   preferences:  uses the synthetic weights of the earlier reference instance.
# Run from lib/CorleoneGame with julia --project=. library/tupelo/tupelo_allowance.jl.
using CorleoneGame
isdefined(@__MODULE__, :Tupelo) || include(joinpath(@__DIR__, "tupelo.jl"))
module TupeloAllowance
    using CorleoneGame
    using ..Tupelo

    parameters() = Tupelo.Parameters(;
        b = 0.55, muE = 0.025, muH = 0.1, dE = 0.05, dH = 1.2, kappa = 0.65,
        p_decay = 0.2, q_decay = 0.06, w_A = (1 / 1.23, 0.08 / 1.23, 0.15 / 1.23, 0.0),
        w_B = (1 / 1.025, 0.025 / 1.025, 0.0), U_A = 2.0, U_B = 3.5, C_A = 1.0,
        saturation = true, photoperiod = false
    )

    function main(; folder = joinpath(@__DIR__, "output_allowance"))
        mkpath(folder)
        # Reuse the reference observations so that forward.png compares this
        # candidate with the synthetic data generated from the reference instance.
        observations = joinpath(@__DIR__, "output", "tupelo_synthetic.csv")
        isfile(observations) && cp(observations, joinpath(folder, "tupelo_synthetic.csv"); force = true)
        options = Tupelo.algorithm_options(; damping = 0.5)  # setting of the earlier reference record
        return run_game(Tupelo.game(parameters()); options, folder)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    mkpath(joinpath(@__DIR__, "output_allowance"))
    CorleoneGame.with_run_log(joinpath(@__DIR__, "output_allowance", "run.log")) do
        TupeloAllowance.main()
    end
end
