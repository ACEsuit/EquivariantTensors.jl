# ACEradials ↔ Polynomials4ML — Reorganization Notes

Working notes (2026-06-13). Analysis only — no code changed. Goal: read
through both packages, catalogue overlapping/duplicated functionality, and
record where things *should* live. Companion to `agents/radials.md` (the
original ACEradials port plan) and `agents/restructure.md` §10 (which parks
the "ACEradials ↔ P4ML spline ownership" question that triggered this).

Versions inspected: ACEradials @ `lib/ACEradials` (this repo, branch
`restruct_atomsdoc`); Polynomials4ML v0.5.9 (resolved in the project
Manifest, depot dir `…/Polynomials4ML/i6U5x`).

---

## 1. The two packages, and the line between them

**Polynomials4ML (P4ML)** — a *general-purpose, package-agnostic* ML basis
library. One abstraction: `AbstractP4MLBasis <: AbstractLuxLayer`, mapping a
low-dimensional input (scalar or `SVector`) to a vector of features. It owns
the *machinery* that makes a basis usable:

- a unified batched, in-place interface (`evaluate!`, `evaluate_ed!`) with a
  single-input → `StaticBatch` redirect (`interface.jl`);
- CPU SIMD paths **and** KernelAbstractions/GPU paths, dispatched on array
  type (`AbstractGPUArray` → `ka_evaluate!`);
- WithAlloc/Bumper allocation (`whatalloc`, `_valtype`/`_gradtype`);
- generic ChainRules `rrule`s (`generic_ad.jl`);
- the concrete bases: orthogonal polys (legendre/jacobi/chebyshev/cheb/
  monomials/bernstein), trig (`ctrig`/`rtrig`), spherical harmonics
  (`sphericart`), atomic orbitals (`atomicorbitals/`, Gaussian/Slater
  decay), discrete weights;
- **cubic splines** (`splinify.jl`): `CubicSplines <: AbstractP4MLBasis` plus
  `splinify(f, x0, x1, NX)` that interpolates *any* univariate-input basis;
- composition helpers: `TransformedBasis` (input-transform ∘ basis) and
  `WrappedBasis` (turn an arbitrary Lux layer into a P4ML basis).

Crucially **P4ML depends only on LuxCore + ACEbase** — not Lux, not
DecoratedParticles, not AtomsBase. Anything pushed "up" into P4ML must respect
that: pure, low-dimensional numeric kernels only.

**ACEradials** — *ACE-specific* radial-basis assembly (`Rnl`). It owns the
domain logic P4ML deliberately doesn't:

- `LearnableRnlBasis` / `SplineRnlBasis`, both multi-species via
  `SMatrix{NZ,NZ}` of transforms / envelopes / (splines), a per-`(n,l)` spec,
  and a per-species `Wnlq` weight tensor;
- the calling convention `evaluate(basis, r, Zi, Zj, ps, st)` (+ `_batched`,
  `_ed`, `_ed_batched`), with ForwardDiff-dual `evaluate_ed` and a hand-rolled
  `Wnlq` pullback;
- `splinify(::LearnableRnlBasis)` → `SplineRnlBasis` (delegates the actual
  interpolation to `P4ML.splinify`/`P4ML.CubicSplines`);
- envelopes `PolyEnvelope1sR` / `PolyEnvelope2sX` (interface
  `evaluate(env, r, x)`);
- transforms: a scalar Agnesi (`transforms.jl`) **and** a species-pair
  `DPTransform` Agnesi (`agnesi_dp.jl`);
- `elements.jl` (z↔i helpers, `bond_len` stub via AtomsBaseExt);
- `transsplines.jl` (`TransSelSplines` / `trans_splines`) — just moved here
  from ET in PR #118; a GPU spline radial basis built on `EmbedDP` +
  `SelectLinL`.

Dependency direction (all acyclic): `ACEradials → ET → P4ML` and
`ACEradials → P4ML`. So "move it up into P4ML" is always dependency-legal;
"move it up into ET" is legal for ACE-agnostic pieces that still need ET's
particle/graph machinery.

---

## 2. Duplicated / overlapping functionality

### 2.1 Cubic-spline evaluation — **three** code paths, two kernel copies

This is the big one, and the reason `agents/restructure.md` §10 flagged it.

| Path | Storage | Scalar math | Batched eval |
|------|---------|-------------|--------------|
| **P4ML** `splinify.jl` | `CubicSplines{NX,NU,T}` (`SVector` F, G) | `_eval_cubic`, `_eval_cubspl`, `_cubspl_widthgrad` | CPU SIMD `_evaluate!` + KA `_ka_evaluate!` |
| **ACEradials** `Rnl_splines.jl` + `splinify.jl` | `SMatrix{NZ,NZ}` of `P4ML.CubicSplines` | *reuses* `P4ML._eval_cubspl` | bespoke per-species loop (CPU only) |
| **ACEradials** `transsplines.jl` | own `refstate` (F, G, x0, x1 re-packed from `P4ML._init_luxstate`) | **own copies** `_eval_cubic`, `_spl_grid`, `_eval_cubic_widthgrad` | own KA kernels `_etspl_kernel!` / `_etspl_ed_kernel!` |

- `_eval_cubic` exists **verbatim** in P4ML `splinify.jl:103` and ACEradials
  `transsplines.jl:225`. `_spl_grid` (ACEradials) and the grid arithmetic
  inside `_eval_cubspl` (P4ML) are the same computation written twice.
- The file header of `transsplines.jl` is explicit: *"essentially complete
  re-implementation of P4ML spline evaluation because some small details just
  don't run nice on GPU. TODO (1) check whether this code can be unified
  between P4ML and ET to reduce duplication."* And P4ML's `splinify.jl` TODO
  is the mirror image: *"consider allowing a coordinate transformation before
  the splines and a post-multiplication with an envelope … very useful for
  ACE applications … but might be better to implement as wrappers."*
- So both sides already know the duplication exists and roughly how to fix it.
  The blocker has been that P4ML's scalar helpers are unexported `_`-prefixed
  internals and (per the comment) not GPU-clean, so transsplines forked them.

### 2.2 Agnesi transform — two copies *inside ACEradials*

`transforms.jl` (`GeneralizedAgnesiTransform` + `NormalizedTransform`, scalar
`r`) and `agnesi_dp.jl` (`agnesi_params` / `eval_agnesi` / 6-arg
`agnesi_transform`, species-pair `DPTransform`) implement the same maths with
different calling conventions. Both files carry a comment pointing at
`radials.md` §4 promising unification "in a later pass". (A third, ET-side
copy was the origin of `agnesi_dp.jl`; that one is already gone.)

### 2.3 Forward-mode `evaluate_ed` for low-dim → high-dim

The ForwardDiff-`Dual` "value + derivative" pattern is re-implemented in:
P4ML `interface.jl`/`transformed.jl`/`wrappedbasis.jl`, and ACEradials
`Rnl_learnable.jl` / `Rnl_splines.jl`. Same idea each time; P4ML's is the
generic one.

### 2.4 Transform-∘-basis composition

`P4ML.TransformedBasis` (trans ∘ basis), ET's `EmbedDP` (trans → basis →
post), and `LearnableRnlBasis` (transform applied inside `evaluate`) all
express the same `r → x → features` pipeline. `TransSelSplines` is
`trans → spline → ·envelope` — i.e. `TransformedBasis` + a categorical spline
selection + an envelope post-multiply.

### 2.5 Categorical selection of a weight slab

`Wnlq[:, :, iz, jz]` (ACEradials, select by species pair) and ET's
`SelectLinL` (`W[:, :, selector(x)] * P`, select by an arbitrary category)
are the same operation — pick a weight matrix by a categorical index — with
different storage and separate KA/rrule code. `TransSelSplines`' `selector`
is the same pattern applied to *which spline* to evaluate.

---

## 3. Where things should live

Guiding rule: **pure low-dimensional numeric kernels → P4ML; ACE/chemistry/
particle semantics → ACEradials (or ET).** P4ML must stay LuxCore-only.

### 3.1 Splines: make P4ML the single source of the cubic-spline *math*

The cubic-spline evaluation math (`_eval_cubic`, grid arithmetic, the
dual-number width-gradient, the basic KA kernel) is exactly the kind of pure
numeric kernel that belongs in P4ML, and P4ML already hosts it for
`CubicSplines`. The duplication in `transsplines.jl` should be removed by
having ACEradials *call* P4ML's helpers rather than forking them.

**DECISION (2026-06-13, CO): Option B is selected.** Option A stays parked
as the possible longer-term convergence point (§3.4), to revisit only if a
second consumer of "selected splines" appears.

Two structurally different end-states:

- **Option A — push the whole "selected, transformed, enveloped spline" into
  P4ML.** Add to P4ML's `CubicSplines` (or a new `SelectedSplines` basis) an
  optional pre-transform, a per-input *integer* category selector, and an
  envelope post-multiply. Pro: one GPU spline implementation, ACEradials
  becomes pure assembly. Con: these are the "a bit specific for a general
  purpose library" features P4ML's own TODO hesitated over; selection-by-`z`
  and the `DPTransform`/`PState` parts **cannot** go (DP/AtomsBase deps), so
  only a generic Int-indexed selector is admissible upstream.

- **Option B (recommended near-term) — P4ML owns the kernel math; ACEradials
  owns the orchestration.** Promote P4ML's scalar spline helpers
  (`_eval_cubic`, `_eval_cubspl`, `_cubspl_widthgrad`) to a stable,
  `@inline`, GPU-safe, (semi-)public surface, and *fix the GPU "details"
  there once*. `TransSelSplines` then keeps its selection + transform +
  envelope orchestration (the genuinely ACE-specific part) but drops its
  private `_eval_cubic`/`_spl_grid`/`_eval_cubic_widthgrad` and calls the
  P4ML versions inside its KA kernels. Result: exactly one `_eval_cubic` in
  the ecosystem, GPU-correct, with no new ACE-flavoured surface forced into
  P4ML.

  Prerequisite/cost: P4ML's helpers are currently unexported internals whose
  GPU-cleanliness is exactly what was in doubt — so step 1 is really *"harden
  P4ML's scalar spline helpers for GPU and commit to them as an API,"* then
  delete the forks. This is the smallest change that kills the duplication
  and is reversible toward Option A later if a second consumer of "selected
  splines" appears.

Either way the **three paths collapse to one storage + one math kernel**:
`P4ML.CubicSplines` is the storage/math; `SplineRnlBasis` already reuses
`_eval_cubspl` (keep as-is); `TransSelSplines` stops forking and reuses the
same helpers.

### 3.2 Agnesi: collapse the two ACEradials copies (internal, independent)

Pick the scalar `GeneralizedAgnesiTransform`/`NormalizedTransform` as the
primitive and make the `DPTransform` version (`agnesi_dp.jl`) a thin adapter
that evaluates the scalar transform on `norm(x.𝐫)` with species-pair
parameter lookup — or vice versa. This is pure ACEradials cleanup, needs no
P4ML change, and can be done before/independently of the spline work. Settles
the `radials.md` §4 / code-comment promise.

### 3.3 Envelopes: keep in ACEradials

`PolyEnvelope1sR`/`2sX` are ACE radial-cutoff semantics — keep them in
ACEradials. If Option A is ever taken, the envelope *multiply* would move into
the P4ML spline basis, but the envelope *types* still belong to ACEradials.
For Option B, the envelope stays a post-multiply applied by the ACEradials
layer (as it is today).

### 3.4 Composition + selection abstractions: note, don't act yet

`TransformedBasis` (P4ML) vs `EmbedDP` (ET) and `Wnlq`-selection vs
`SelectLinL` are real conceptual duplicates, but they sit at the
ET/P4ML boundary and serve live code on both sides. Deferred: revisit once the
spline consolidation lands, since `TransSelSplines` is the concrete place all
three (transform-chain, categorical selection, spline) meet — if it ends up
expressed as `EmbedDP(trans, CubicSplines, post)` that would retire the
bespoke layer and is the natural Option-A convergence point. Don't refactor
these speculatively.

---

## 4. Suggested sequencing

1. **ACEradials-internal Agnesi merge** (§3.2) — self-contained, no upstream
   dependency, clears a long-standing TODO. Low risk.
2. **Harden P4ML scalar spline helpers for GPU + make them API** (§3.1, the
   P4ML-side prerequisite) — upstream PR to Polynomials4ML.
3. **De-fork `transsplines.jl`** onto the P4ML helpers (§3.1 Option B) — one
   `_eval_cubic` left in the ecosystem. Verify GPU path (the commented-out
   Metal block in `test_splines.jl`).
4. **Re-evaluate Option A** only if/when a second "selected spline" consumer
   appears, or if `EmbedDP`/`TransformedBasis` are unified for other reasons.

Steps 2–3 are the payload of the `restructure.md` §10 "spline ownership"
item; step 1 closes `radials.md` §4; step 4 stays parked.

---

## 5. Constraints to respect

- **P4ML stays LuxCore-only** — no DecoratedParticles / AtomsBase / Lux. Only
  pure numeric kernels and generic (Int-indexed) selection may move up.
- GPU paths are KernelAbstractions; helpers shared into kernels must be
  `@inline` and free of CPU-only constructs (this is precisely why the fork
  happened — fixing it upstream is the point).
- Don't hard-code floating-point types in any moved kernel (CLAUDE.md).
- The `evaluate(basis, r, Zi, Zj, ps, st)` calling convention of the `Rnl`
  bases is depended on by ACE models downstream — preserve it.
