# Root-package tests

Run from the repository root. Each named group uses its own dependency environment:

```sh
GROUP=Core julia --project=. -e 'using Pkg; Pkg.test()'
GROUP=Examples julia --project=. -e 'using Pkg; Pkg.test()'
GROUP=Plotting julia --project=. -e 'using Pkg; Pkg.test()'
GROUP=QA julia --project=. -e 'using Pkg; Pkg.test()'
GROUP=Differentiation julia --project=. -e 'using Pkg; Pkg.test()'
GROUP=Distributed julia --project=. -e 'using Pkg; Pkg.test()'
```

The default `All` selection runs Core. The other groups have separate CI jobs.
Set `JULIA_NUM_THREADS=2` or higher to exercise threaded execution with multiple
threads.

Core covers annotations, bounds, reset schedules, trajectories, optimization
contracts, the Lux interface, state handoff, initialization from reference solves,
and focused AD regressions. Initialization uses annotation defaults and explicit
shooting initialization helpers. Legacy whole-layer constructor callbacks remain
unsupported. Deprecated constructors have their own translation tests.

Differentiation checks full ODE solves against an analytic loss and gradient at
two inputs. It covers shared parameters, free global initial values, interior
matching residuals, and differentiable numerical values held in Lux state.
ForwardDiff and Zygote exercise serial and threaded layers. Zygote's reverse rule
records the block solves serially. ReverseDiff uses serial execution with
`FullSpecialize` and `SensitivityADPassThrough()`; compiled tapes are not covered.
Mooncake and Enzyme are not part of this regression group.

Distributed starts two local worker processes, loads the required packages on
them, and removes them in a `finally` block. It checks repeated forward calls and
ForwardDiff gradients against serial execution and finite differences. The named
objective and model definitions are loaded on every worker to preserve
ForwardDiff tag types; anonymous objective closures can fail after serialization.
It needs
permission to open localhost sockets. It does not establish parallel reverse AD
support.
