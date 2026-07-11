# ACEradials — design notes & remaining work

Notes for `lib/ACEradials` (the ACE-specific radial-basis subdir package). For
the current structure and where it sits in the ecosystem, see
`docs/src/architecture.md`. Companions: `agents/restructure.md` (ET core) and
`agents/initializers.md`. (This file absorbs the former
`radials_restructure.md`, whose spline/Agnesi consolidation has landed.)

## Status — what's done

- **Port** of the canonical learnable/splined `Rnl` bases from
  `ACEpotentials.jl/src/models/` into `lib/ACEradials`: types renamed
  (`LearnableRnlrzzBasis`→`LearnableRnlBasis`, `SplineRnlrzzBasis`→
  `SplineRnlBasis`); `_i2z`/`_z2i`/`_get_nz` helpers local; splines stored as
  `Polynomials4ML.CubicSplines` (dropping the `Interpolations`/`OffsetArrays`
  deps). Plus envelopes, scalar + species-pair Agnesi transforms, `splinify`,
  `TransSelSplines`, and `learnable_Rnl_basis`.
- **Spline/Agnesi consolidation**: `agnesi_dp.jl` is now a thin adapter over
  the scalar `GeneralizedAgnesiTransform`/`NormalizedTransform` (#119); P4ML
  owns the GPU-safe cubic-spline kernel (#122, P4ML v0.5.10) and
  `TransSelSplines` calls it (de-fork #120) — one `_eval_cubic` in the
  ecosystem.

## Decision record — why a subdir package

ACEradials lives in `lib/ACEradials` as a separately registered, separately
versioned subdir package (the Lux.jl `lib/` model), depending on ET. Rather
than an ET submodule (radials churn would force ET breaking bumps on all
dependents) or a separate repo (the ET-coupled features below are
co-evolution work, cheaper as atomic in-repo PRs). Graduation path: once the
API stabilises it can move to its own repo with the same UUID/name
(non-breaking); the reverse would be breaking, so this layout preserves
optionality. The `evaluate(basis, r, Zi, Zj, ps, st)` calling convention is
depended on by downstream ACE models — preserve it.

## Remaining future work

**ACEpotentials adaptation (Stage 2 — separate later PR; repo not in this
workspace):** remove the old `src/models/{radial_envelopes,radial_transforms,
Rnl_basis,Rnl_learnable,Rnl_splines}.jl`; replace with re-exports +
backward-compat aliases (`LearnableRnlrzzBasis = LearnableRnlBasis`, etc.);
keep `ace_heuristics.jl` a thin wrapper over `learnable_Rnl_basis`; tighten the
ET/ACEradials version bounds; run the ACEpotentials suite.

**Additional functionality (parked, triage when needed):**
- Learnable envelope parameters (`p` in `PolyEnvelope2sX` as a Lux param); a
  soft distance-based envelope `(cos(πr/rcut)+1)/2`.
- Unified scalar/edge transform: one abstraction callable both as
  `t(r, iz, jz)` (scalar, for `LearnableRnlBasis`) and on an XState edge (for
  `EmbedDP`), sharing parameter storage — removes the remaining scalar-vs-DP
  transform split. Plus learnable transform params (`r0, a, p, q`).
- `orthonormal_Rnl_basis` constructor (orthonormalisation code landed in ET
  PR #105); `pair_Rnl_basis(...)` convenience (the `l = 0` restriction).
- Move `sparse_AA_spec` + level functions (`TotalDegree`, `SparseLevelSpec`)
  and the smoothness priors (`algebraic/gaussian/exp_smoothness_prior`) from
  ACEpotentials into ET — architecture-level, not potentials-specific.
- GPU path: a `to_TransSelSplines(spl::SplineRnlBasis, selector)` converter to
  migrate the CPU spline basis onto the KA-backed `TransSelSplines`.

**Parked spline decisions (revisit only with a second consumer):**
- *Option A* — push the whole "selected, transformed, enveloped spline" into
  P4ML (a generic Int-indexed `SelectedSplines` basis). Deferred in favour of
  the Option B that landed (P4ML owns the kernel math, ACEradials owns the
  orchestration). Take Option A only if a second "selected spline" consumer
  appears.
- *Composition/selection unification* — `TransformedBasis` (P4ML) vs `EmbedDP`
  (ET) vs `LearnableRnlBasis`, and `Wnlq`-selection vs `SelectLinL`, are real
  conceptual duplicates that meet in `TransSelSplines`. If it can be expressed
  as `EmbedDP(trans, CubicSplines, post)` the bespoke layer retires (the
  Option-A convergence point). Don't refactor speculatively.

The `Wnlq * P(x)` channel-mixing vs ET's equivariant linear layer, and the
"fold" converter that absorbs a trained ET mixing into the radial basis, are
recorded in `agents/restructure.md` §6.1.
