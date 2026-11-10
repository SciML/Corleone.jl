# Available problems

The registry contains the following constructors. Links open the implemented
dynamics, objective, bounds, and constraints. The MINTOC links provide related
mathematical formulations and reference solutions. Implemented control bounds,
horizons, or continuous relaxations may differ; use the source definition when
interpreting a benchmark result. Horizons are in each model's
physical time coordinate; the input grids remain normalized to `[0, 1]`.

| Registry identifier | Controls | Time span | Related formulation |
|:--|--:|:--|:--|
| [`batch_reactor`](https://github.com/SciML/Corleone.jl/blob/main/lib/OptimalControlBenchmarks/src/problems/batch_reactor.jl) | 1 | `(0.0, 1.0)` | [MINTOC](https://mintoc.de/index.php?title=Batch_reactor) |
| [`bryson_denham`](https://github.com/SciML/Corleone.jl/blob/main/lib/OptimalControlBenchmarks/src/problems/bryson_denham.jl) | 1 | `(0.0, 1.0)` | [MINTOC](https://mintoc.de/index.php?title=Bryson_Denham) |
| [`cart_pendulum`](https://github.com/SciML/Corleone.jl/blob/main/lib/OptimalControlBenchmarks/src/problems/cart_pendulum.jl) | 1 | `(0.0, 4.0)` | [MINTOC](https://mintoc.de/index.php?title=Cart_Pendulum) |
| [`catalyst_mixing`](https://github.com/SciML/Corleone.jl/blob/main/lib/OptimalControlBenchmarks/src/problems/catalyst_mixing.jl) | 1 | `(0.0, 1.0)` | [MINTOC](https://mintoc.de/index.php?title=Catalyst_mixing_problem) |
| [`cushioned_oscillation`](https://github.com/SciML/Corleone.jl/blob/main/lib/OptimalControlBenchmarks/src/problems/cushioned_oscillation.jl) | 1 | `(0.0, 10.0)` | [MINTOC](https://mintoc.de/index.php?title=Cushioned_Oscillation) |
| [`denbigh`](https://github.com/SciML/Corleone.jl/blob/main/lib/OptimalControlBenchmarks/src/problems/denbigh.jl) | 1 | `(0.0, 1000.0)` | [MINTOC](https://mintoc.de/index.php?title=Denbigh_Reaction) |
| [`dielectrophoretic_particle`](https://github.com/SciML/Corleone.jl/blob/main/lib/OptimalControlBenchmarks/src/problems/dielectrophoretic_particle.jl) | 1 | `(0.0, 5.0)` | [MINTOC](https://mintoc.de/index.php?title=Dielectrophoretic_Particle) |
| [`double_oscillator`](https://github.com/SciML/Corleone.jl/blob/main/lib/OptimalControlBenchmarks/src/problems/double_oscillator.jl) | 1 | `(0.0, 2 * pi)` | [MINTOC](https://mintoc.de/index.php?title=Double_Oscillator) |
| [`ducted_fan`](https://github.com/SciML/Corleone.jl/blob/main/lib/OptimalControlBenchmarks/src/problems/ducted_fan.jl) | 2 | `(0.0, 10.0)` | [MINTOC](https://mintoc.de/index.php?title=Ducted_Fan) |
| [`egerstedt`](https://github.com/SciML/Corleone.jl/blob/main/lib/OptimalControlBenchmarks/src/problems/egerstedt.jl) | 3 | `(0.0, 1.0)` | [MINTOC](https://mintoc.de/index.php?title=Egerstedt_standard_problem) |
| [`electric_car`](https://github.com/SciML/Corleone.jl/blob/main/lib/OptimalControlBenchmarks/src/problems/electric_car.jl) | 1 | `(0.0, 10.0)` | [MINTOC](https://mintoc.de/index.php?title=Electric_Car) |
| [`f8`](https://github.com/SciML/Corleone.jl/blob/main/lib/OptimalControlBenchmarks/src/problems/f8.jl) | 1 | `(0.0, 10.0)` | [MINTOC](https://mintoc.de/index.php?title=F-8_aircraft) |
| [`fuller`](https://github.com/SciML/Corleone.jl/blob/main/lib/OptimalControlBenchmarks/src/problems/fuller.jl) | 1 | `(0.0, 1.0)` | [MINTOC](https://mintoc.de/index.php?title=Fuller%27s_initial_value_problem) |
| [`goddarts_rocket`](https://github.com/SciML/Corleone.jl/blob/main/lib/OptimalControlBenchmarks/src/problems/goddarts_rocket.jl) | 1 | `(0.0, 10.0)` | [MINTOC](https://mintoc.de/index.php?title=Goddart%27s_rocket_problem) |
| [`hang_glider`](https://github.com/SciML/Corleone.jl/blob/main/lib/OptimalControlBenchmarks/src/problems/hang_glider.jl) | 1 | `(0.0, 100.0)` | [MINTOC](https://mintoc.de/index.php?title=Hang_Glider) |
| [`hanging_chain`](https://github.com/SciML/Corleone.jl/blob/main/lib/OptimalControlBenchmarks/src/problems/hanging_chain.jl) | 1 | `(0.0, 1.0)` | [MINTOC](https://mintoc.de/index.php?title=Hanging_chain_problem) |
| [`jackson`](https://github.com/SciML/Corleone.jl/blob/main/lib/OptimalControlBenchmarks/src/problems/jackson.jl) | 1 | `(0.0, 1.0)` | [MINTOC](https://mintoc.de/index.php?title=Jackson) |
| [`lotka_competitive`](https://github.com/SciML/Corleone.jl/blob/main/lib/OptimalControlBenchmarks/src/problems/lotka_competitive.jl) | 1 | `(0.0, 40.0)` | [MINTOC](https://mintoc.de/index.php?title=LV_Competitive) |
| [`lotka_shared`](https://github.com/SciML/Corleone.jl/blob/main/lib/OptimalControlBenchmarks/src/problems/lotka_shared.jl) | 1 | `(0.0, 40.0)` | [MINTOC](https://mintoc.de/index.php?title=LV_Shared_Resource) |
| [`lotka_volterra`](https://github.com/SciML/Corleone.jl/blob/main/lib/OptimalControlBenchmarks/src/problems/lotka_volterra.jl) | 1 | `(0.0, 12.0)` | [MINTOC](https://mintoc.de/index.php?title=Lotka_Volterra_fishing_problem) |
| [`lqr`](https://github.com/SciML/Corleone.jl/blob/main/lib/OptimalControlBenchmarks/src/problems/lqr.jl) | 1 | `(0.0, 10.0)` | [MINTOC](https://mintoc.de/index.php?title=Linear_Quadratic_Regulator) |
| [`moon_landing`](https://github.com/SciML/Corleone.jl/blob/main/lib/OptimalControlBenchmarks/src/problems/moon_landing.jl) | 1 | `(0.0, 1.0)` | [MINTOC](https://mintoc.de/index.php?title=Moon_Landing) |
| [`mountain_car`](https://github.com/SciML/Corleone.jl/blob/main/lib/OptimalControlBenchmarks/src/problems/mountain_car.jl) | 1 | `(0.0, 100.0)` | [MINTOC](https://mintoc.de/index.php?title=Mountain_Car) |
| [`ocean`](https://github.com/SciML/Corleone.jl/blob/main/lib/OptimalControlBenchmarks/src/problems/ocean.jl) | 2 | `(0.0, 400.0)` | [MINTOC](https://mintoc.de/index.php?title=Ocean) |
| [`particle_steering`](https://github.com/SciML/Corleone.jl/blob/main/lib/OptimalControlBenchmarks/src/problems/particle_steering.jl) | 1 | `(0.0, 10.0)` | [MINTOC](https://mintoc.de/index.php?title=Particle_steering_problem) |
| [`quadrotor`](https://github.com/SciML/Corleone.jl/blob/main/lib/OptimalControlBenchmarks/src/problems/quadrotor.jl) | 4 | `(0.0, 7.5)` | [MINTOC](https://mintoc.de/index.php?title=Quadrotor_helicopter_control_problem) |
| [`rao_mease`](https://github.com/SciML/Corleone.jl/blob/main/lib/OptimalControlBenchmarks/src/problems/rao_mease.jl) | 1 | `(0.0, 10.0)` | [MINTOC](https://mintoc.de/index.php?title=Rao_Mease) |
| [`robbins`](https://github.com/SciML/Corleone.jl/blob/main/lib/OptimalControlBenchmarks/src/problems/robbins.jl) | 1 | `(0.0, 10.0)` | [MINTOC](https://mintoc.de/index.php?title=Robbins) |
| [`three_tank`](https://github.com/SciML/Corleone.jl/blob/main/lib/OptimalControlBenchmarks/src/problems/three_tank.jl) | 3 | `(0.0, 12.0)` | [MINTOC](https://mintoc.de/index.php?title=Three_Tank_multimode_problem) |
| [`tubular_reactor`](https://github.com/SciML/Corleone.jl/blob/main/lib/OptimalControlBenchmarks/src/problems/tubular_reactor.jl) | 1 | `(0.0, 1.0)` | [MINTOC](https://mintoc.de/index.php?title=Tubular_Reactor) |
| [`van_der_pol`](https://github.com/SciML/Corleone.jl/blob/main/lib/OptimalControlBenchmarks/src/problems/van_der_pol.jl) | 1 | `(0.0, 10.0)` | [MINTOC](https://mintoc.de/index.php?title=Van_der_Pol_Oscillator) |

Select a subset by its registry identifiers before calling `run_all`:

```julia
using OptimalControlBenchmarks

catalog = load_benchmarks()
selected = filter(problem -> nameof(problem) in (:lqr, :van_der_pol), catalog)
```

Start with `lqr` to check the solver setup. Use the linked definitions to choose
other models and suitable grids; the shared driver does not guarantee that its
default initial guess or optimizer settings solve every problem.

