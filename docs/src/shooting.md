# [Constructing shooting problems](@id shooting_formulation)

A shooting method replaces a continuous dynamic optimization problem with an
optimization over parameter values and, optionally, intermediate state values.
Corleone integrates each segment with a SciML solver. The optimization model
combines the resulting trajectory with an objective and constraint callbacks.

## Dynamics and parameter schedules

Consider the ODE

```math
\dot{x}(t)=f(x(t),p(t),t),\qquad
x(t)\in\mathbb{R}^{n_x},\quad p(t)\in\mathbb{R}^{n_p},\quad t\in[t_0,t_f].
```

Here ``x`` denotes the numerical state. The parameter vector ``p`` includes any
control inputs. Julia's ODE interface conventionally calls the state argument
`u`; `trajectory.u` therefore stores samples of ``x``, not just control inputs.

Construct an `ODEProblem`, choose a solver, and pass `annotation => grid` pairs to
[`ShootingLayer`](@ref), as in [Getting Started](getting_started.md). Parameter
switches split the integration without resetting states. The
[scheduling guide](scheduling.md) defines interval values, defaults, and storage.

The integration grid is the sorted union of problem endpoints, parameter switches,
and state resets within the span. Prepared expressions may add evaluation times.
Write it as ``t_0=\tau_0<\cdots<\tau_N=t_f``. Setup caches it in `st.timestops`;
see [schedule changes](scheduling.md#Defaults-and-schedule-changes) before modifying grids.

## Single and multiple shooting

With no interior state resets, each segment starts from the preceding endpoint.
Parameter switches still split integration, but they do not create independent
shooting blocks or state continuity constraints. This is single shooting.

For multiple shooting, select state components at interior times and introduce
independent assigned values ``s_{i,k}``. At time ``\tau_k``, an assigned component
starts the next segment at ``x_i(\tau_k^+)=s_{i,k}``. Other components pass through
the incoming endpoint. For an active **tunable** assignment, Corleone forms

```math
r_{i,k}=s_{i,k}-x_i(\tau_k^-).
```

An optimizer enforces ``r_{i,k}=0`` to recover continuity. Before optimization,
the trajectory may contain jumps. A full-state reset permits an independent
block solve; a partial reset retains sequential dependence for the remaining
components.

| Assignment | Meaning |
|---|---|
| No reset | Pass through the incoming state. |
| Tunable assignment at ``t_0`` | Free initial value, with no continuity equality to the original `u0`. |
| Tunable interior assignment | Shooting value with an assigned-minus-incoming residual. |
| Constant interior assignment | Prescribed reset, with no matching residual. |
| Assignment at ``t_f`` | No outgoing segment or shooting residual. |

[`shooting_constraints`](@ref) flattens state residuals in chronological order,
then state-index order. [`get_number_of_shooting_constraints`](@ref) counts them
from the cached schedules without solving. Parameter residuals follow, as defined below.

## Shared parameters and inserted intervals

Lifting states does not automatically split parameter intervals. The existing
parameter values remain shared across all segments where they are active.

If independent parameter storage is required at full-state shooting boundaries,
call [`add_parameter_segments!`](@ref) after setup. Newly inserted right-hand
intervals duplicate the existing value. For an inserted interval index ``\ell``,
the matching residual is

```math
d_{j,\ell}=p_{j,\ell}-p_{j,\ell-1}.
```

These residuals follow state residuals, in canonical parameter order and then
increasing inserted interval index. Original parameter switches are unconstrained;
fixed parameters do not acquire matching residuals. The layer computes these differences on demand from `ps` and
`st`, rather than storing them in `trajectory.residuals`.

## Integral costs and quadrature states

A running cost can be represented by a quadrature state:

```math
J=\Phi(x(t_f),p(t_f)) + \int_{t_0}^{t_f}L(x(t),p(t),t)\,dt,
\qquad \dot{q}=L(x,p,t),\quad q(t_0)=0.
```

Then ``J=\Phi(x(t_f),p(t_f))+q(t_f)``. A numerical model supplies this additional
state explicitly and can mark it as a quadrature. The ModelingToolkit frontend
introduces quadratures for supported definite integrals.

When a quadrature is lifted, its continuity equality preserves the accumulated
cost from earlier segments. Its terminal value represents the whole horizon
only when those matching constraints are satisfied. Quadrature metadata alone
does not add a differential equation. See the [LQR tutorial](@ref linear_quad_reg)
and [ModelingToolkit tutorial](@ref mtk) for both constructions.

## Bounds and numerical accuracy

Parameter bounds apply to stored interval values. State annotation bounds apply
to tunable initial or shooting values; they are not continuous path constraints.
User constraint callbacks and expressions constrain the samples they evaluate.
Neither finite sampling nor shooting-value bounds certify a condition between
samples.

The ODE solver's tolerances control integration error; the optimizer's tolerances
control its stopping criteria and constraint feasibility. A successful optimization
return code alone does not establish a global optimum or a verified continuous
solution. Inspect constraint residuals and, where useful, replay the optimized
controls without lifted resets, as in the [LTI tutorial](@ref mtk_lti).
