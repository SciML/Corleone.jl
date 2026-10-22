# CorleoneGame benchmark library

This directory contains the 25 biological dynamic games of the CorleoneGame
benchmark library, described in the preprint

> Sager, Plate, Martensen, Ohl, *CorleoneGame: A library of biological dynamic
> games with a companion Julia solver*, arXiv:2610.06307,
> <http://arxiv.org/abs/2610.06307>.

The preprint gives the biological background, equations, parameters, sources, and
an interpretation of the computed equilibria for every case.

## Contents and usage

Each case has its own directory `CASE/` with the case file `CASE.jl` and the saved
reference results in `CASE/output/`: `trajectory.csv`, `controls.csv`,
`iterations.csv`, `iterations/controls_n*_iter*.csv` (damped profile of every
best-response round), `report.txt`, and `run.log`. Plots are not stored; they are
redrawn from the CSVs with `scripts/plot_library.jl` (see the
[package README](../README.md)). Run a case from `lib/CorleoneGame` with

```bash
julia --project=. library/CASE/CASE.jl [single|multiple|synthetic]
```

Every case module defines `Parameters` (model parameters, objective weights,
restrictions), `algorithm_options` (solver settings), `game`, and `main`.
Objective weights and restrictions can be changed through `Parameters`; a
restriction set to `nothing` is disabled. All cases use 10 coarse and 40 refined
control intervals by default. The models are synthetic or uncalibrated
benchmarks motivated by the cited literature, unless stated otherwise.

The first entry of the category line at the top of each case file gives the
temporal class. In *seasonal* cases, one season ends without returning to its
initial state. In *periodical* cases, one cycle ends in a return band around
the initial state. In *episodical* cases, one episode starts from a prescribed
state and ends with a target or at a fixed time. In seven cases
(batchFermentation, circadianCycle, fishery, immuneClearance, predatorPrey,
seedGermination, and woundRepair), player A also chooses the duration.

## Cases

### algae

Carbon-uptake allocation in an algae community (periodical, 4 players). Each
abstract uptake type allocates its uptake machinery between dissolved CO₂ and
bicarbonate from shared resource pools. Over one supply cycle of a maintained
culture, each strain maximizes its gross production while avoiding extreme
specialization; biomasses and resources return to their initial levels.

### antColonies

Competition of ant colonies for food, water, and territory (seasonal, 3 players).
Each colony divides its workers between food foraging, water collection, and
territorial aggression; stored food is converted into brood that later becomes
workers. Effort budgets and a minimum food store restrict the colonies during
one foraging season.

### avoidance

Signaled active avoidance with two decentralized control modules (episodical,
2 players). During one trial block with a warning cue, each module invests effort
to build avoidance readiness, which consumes a shared, slowly recovering
motivational resource. The modules are partially aligned, since each benefits
from the other's readiness.

### batchFermentation

Division of labor in a batch fermentation (episodical, 3 players). An upstream
strain converts substrate into an intermediate that two downstream strains turn
into the product; each strain pays for production with a growth burden. The batch
ends when the product target is reached; the upstream strain chooses its duration.

### biofilm

Matrix investment and dispersal in a biofilm (episodical, 3 players). Each strain
allocates substrate to protective extracellular polymeric substances (EPS) and
chooses when to disperse, while all strains share substrate and oxygen. The
strains differ in how they value attached biomass versus dispersal output.

### ciliate

Bacterium–ciliate predator–prey interaction (periodical, 2 players). The bacteria
command a grazing defense and the ciliates their feeding activity; the expressed
phenotypes follow the commands with a delay. Over one seasonal cycle, both
populations and both phenotypes return to their initial values.

### circadianCycle

The ten-state PER/TIM circadian oscillator of *Drosophila* (periodical,
2 players), with the kinetics of Slaby et al. (2007), Appendix B. Player A
modulates *per* transcription and chooses the cycle length; player B, the
light-input pathway, controls TIM degradation. Both prefer balanced PER/TIM
levels and low intervention effort, and all ten concentrations close over one cycle.

### fishery

Two fleets harvest a renewable logistic fish stock under rotational management
(periodical, 2 players), following the two-harvester model of Suri (2008).
Each fleet maximizes its discounted profit under per-opening catch quotas. The
stock returns to its initial value after one rotation, whose length is chosen by
fleet A.

### foraging

Competitive foraging on a shared, renewable food patch (episodical, 3 players).
Each forager chooses its foraging effort and converts food into energy reserves
and a retained harvest. Effort budgets and a shared patch floor at 70 % of the
carrying capacity restrict one foraging episode.

### immuneClearance

Immune clearance and viral immune evasion during one acute infection
(episodical, 2 players). The host activates cytotoxic and antibody responses and
ends the episode when the viral load reaches a prescribed low level; it thus
chooses the duration. The virus invests in immune evasion to maximize its
accumulated viral load.

### metabolicPathway

Enzyme allocation in a branched metabolic pathway (periodical, 2 players). Two
modules express parallel enzymes that convert a periodically supplied substrate
into a shared product, and each reaction yields a byproduct that inhibits the
module's own reaction. Effort budgets apply, and all pools return after one
supply cycle; the shared product makes this a public-goods game.

### microbial

Microbial cross-feeding in a chemostat (periodical, 10 players). Ten strains,
organized in five reciprocal pairs, compete for a common substrate and secrete
metabolites that their partners require. Each strain chooses its secretion
allocation over one 80 h feed cycle, after which all strains and pools return.
The number of strains is a parameter.

### mitochondria

Mitochondrial maintenance competition (periodical, 2 players). A wild-type and a
variant mtDNA subpopulation share one energetic pool under recurring activity
and invest in biogenesis to keep their own healthy mass high, subject to effort
budgets. All pools return after one activity cycle.

### motor

Physically coupled motor control (episodical, 2 players), following the
joint-action experiments of Chackochan and Sanguineti (2019). Two people move
handles that are connected by a virtual spring toward a common target, each
through a private via-point. Each player controls a two-dimensional muscle
command during one 0.5 s reach; the via-points are fixed-time costs.

### neural

Excitatory–inhibitory regulation in a Wilson–Cowan circuit (periodical,
2 players). Agent A controls the excitatory drive to support activity, agent B
the inhibitory drive to regulate excitation, with different activity targets.
Both activities return after one cycle of the slowly modulated background drive.

### phage

Bacterium–phage arms race within one growth episode (episodical, 2 players).
The bacteria regulate their resistance against a growth cost, and the phage
population adjusts the infectivity of its progeny; both expressed traits follow
the commands with a delay. The model distinguishes susceptible and infected
bacteria, free phage, and both phenotypes.

### plantNitrogen

Competing plants and soil nitrogen (seasonal, 3 players). Each plant controls its
nitrogen uptake effort; nitrogen builds photosynthetic capacity, while the
plants shade one another. Effort budgets and a shared soil-nitrogen floor apply
during one growing season.

### plantWater

Root investment across shared soil layers (seasonal, 4 players). Each plant
allocates root investment across three soil layers during a season with a wet
first half and a dry second half, and the plants specialize on different layers.
A seasonal water-uptake allowance limits each plant.

### pollination

Nectar investment and pollinator visitation (seasonal, 4 players). Two plants
choose nectar production, and two pollinators choose their visitation
intensities on both plants; each pollinator is more efficient on one plant.
Effort budgets and population floors restrict one flowering season.

### predatorPrey

A predator–prey community with three prey and two predator species (periodical,
5 players) that oscillates on its own (paradox of enrichment). Prey invest in
defense and growth, predators in hunting and maintenance. All populations return
after one cycle, whose duration is chosen by prey species A.

### proteinFolding

Chaperone assistance after a stress pulse (episodical, 2 players). Two chaperone
systems with different efficiencies share a limited, slowly restored assistance
capacity and promote folding to the native state. Each system chooses its
assistance effort under a budget; both value the same native pool.

### seedGermination

Germination of three seed cohorts sharing one water pool (seasonal, 3 players).
Each cohort controls embryo activation, which speeds development but consumes
shared water. The window ends when the leading cohort A reaches radicle
emergence; cohort A chooses the duration.

### sexRatio

Sex allocation among foundresses under local mate competition (seasonal,
3 players). Each foundress chooses the male fraction of its offspring over one
reproductive season; sons are more resource-intensive to provision from a shared,
seasonally replenished resource. Budgets on the squared allocation and a floor on
the shared resource apply.

### tupelo

Plant defense and herbivore hatching in the tupelo–leafminer system (seasonal,
2 players), mapped from scenario 11 of Low, Ellner, and Holden (2013). The tree
chooses a defense schedule, and the leafminer chooses the hatching hazard of its
eggs; states are leaf area, eggs, and larvae. `tupelo_allowance.jl` defines the
alternative instance of Section 5 of the preprint with carbon and effort
allowances (results in `output_allowance/`), and `tupelo_synthetic.jl` generates
noisy observations from the saved trajectory.

### woundRepair

Wound repair and remodeling (episodical, 3 players). Inflammatory recruitment,
fibroblast recruitment, and oxygen support are three players that weigh effort,
excess inflammation, and collagen quality differently. The episode ends when the
wound deficit has fallen to a prescribed residual size; player A chooses the
duration.
