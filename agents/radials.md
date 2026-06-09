# Radial Basis Submodule — Implementation Plan

## Context and goals

The goal is to move the canonical radial basis implementation from
`ACEpotentials.jl/src/models/` into EquivariantTensors.jl as a self-contained
submodule `src/radials/`. This makes ET the single source of truth for learnable
Rnl bases, reduces duplication, and lets other packages (not just ACEpotentials)
build ACE-style models on top of ET without taking a dependency on ACEpotentials.

The core design in ACEpotentials is:

```
r  --[transform]--> x ∈ [-1,1]  --[poly basis Pq(x)]--> P
                                                          |
                                           Rnl = Wnlq * (P .* envelope(r, x))
```

- **Learnable** variant (`LearnableRnlrzzBasis`): `Wnlq` is a free Lux parameter.
- **Frozen/splined** variant (`SplineRnlrzzBasis`): the `x -> Rnl(x)` map is
  replaced by cubic splines; `Wnlq` is baked in. No learnable parameters remain.

Both types are multi-species: all fields that are species-dependent are
`SMatrix{NZ,NZ,...}` indexed by `(iz0, iz1)`.

---

## What currently lives where

### Already in ET (`src/transforms/`, `src/embed/`)

| Component | File | Notes |
|-----------|------|-------|
| `DPTransform` / `dp_transform` | `transforms/decpart.jl` | Generic function-as-transform layer |
| `eval_agnesi` / `agnesi_params` / `agnesi_transform` | `transforms/agnesi.jl` | ET version of the Agnesi transform; maps `(r, z0, z1)` to `y ∈ [-1,1]` |
| `EmbedDP` | `embed/embeddings.jl` | `trans -> basis -> post` chain with `evaluate_ed` |
| `SelectLinL` | `utils/selectlinl.jl` | Species-selective linear layer `W[icat] * P` |
| `TransSelSplines` / `trans_splines` | `embed/transsplines.jl` | KA spline evaluation (GPU-ready) |

### In ACEpotentials (`src/models/`)

| Component | File | Notes |
|-----------|------|-------|
| `PolyEnvelope2sX`, `PolyEnvelope1sR`, `ACE1_PolyEnvelope1sR` | `radial_envelopes.jl` | Cutoff envelopes |
| `GeneralizedAgnesiTransform`, `NormalizedTransform` | `radial_transforms.jl` | ACEpotentials-side Agnesi, distinct from ET's version |
| `LearnableRnlrzzBasis` | `Rnl_basis.jl` + `Rnl_learnable.jl` | Learnable Lux layer, per-species `Wnlq` |
| `SplineRnlrzzBasis` | `Rnl_basis.jl` + `Rnl_splines.jl` | Splined, no-parameter variant |
| `splinify` (Models version) | `Rnl_basis.jl` | Converts learnable -> splined |
| `ace_learnable_Rnlrzz` | `ace_heuristics.jl` | High-level constructor with defaults |

The ET-model splinify (`et_models/splinify.jl`) is a separate, parallel
implementation built on `TransSelSplines` — it targets the `EmbedDP`+`SelectLinL`
architecture rather than the `LearnableRnlrzzBasis` one.

---

## Proposed ET submodule: `src/radials/`

### New files

```
src/radials/
    radials.jl          # module file, include list
    envelopes.jl        # cutoff envelope types
    Rnl_learnable.jl    # LearnableRnlBasis (renamed, simplified)
    Rnl_splines.jl      # SplineRnlBasis
    splinify.jl         # splinify(learnable, ps) -> spline
    constructors.jl     # high-level constructor (replaces ace_learnable_Rnlrzz)
```

### Design decisions

**1. Naming and scope**

Rename `LearnableRnlrzzBasis` → `LearnableRnlBasis` and `SplineRnlrzzBasis` →
`SplineRnlBasis`. The `rzz` suffix was an implementation detail (multi-species
storage); the public name shouldn't encode it.

==> agreed 

The `NZ`-species generality is preserved via `SMatrix{NZ,NZ,...}` fields, but
the `_i2z` field (atomic number lookup) and helper functions `_z2i`, `_get_nz`
move into ET's utils.

==> agreed 

**2. Interface**

Keep the existing calling convention:
```julia
evaluate(basis, r, Zi, Zj, ps, st)           # single distance
evaluate_batched(basis, rs, zi, zjs, ps, st) # vector of distances
evaluate_ed(basis, r, Zi, Zj, ps, st)        # value + dr derivative
```
These match what ACE models already call; no interface breakage.

**3. Envelopes**

Move `PolyEnvelope2sX`, `PolyEnvelope1sR`, `ACE1_PolyEnvelope1sR` verbatim from
ACEpotentials into `envelopes.jl`. These are simple structs with `evaluate`
methods; no rework needed.

The envelope interface is `evaluate(env, r, x) -> scalar`, where `r` is the
physical distance and `x = transform(r)` is already computed. This stays as-is.

**4. Transforms**

The Agnesi transform already exists in ET (`transforms/agnesi.jl`) but has a
different calling convention: it takes an `XState` / `NamedTuple` edge descriptor
rather than `(r, iz, jz)` indices. The `LearnableRnlBasis` needs the
ACEpotentials-style `NormalizedTransform` that takes a scalar `r`.

Two options:
- (a) Port `GeneralizedAgnesiTransform` + `NormalizedTransform` into
  `src/radials/` as internal helpers.
- (b) Refactor `agnesi_transform` in `src/transforms/agnesi.jl` to expose a
  scalar `(r, params) -> y` path and reuse it.

**Recommendation: option (a) for the first pass.** Keep the two representations
(the full edge-aware `DPTransform` wrapper and the scalar `r`-only transform)
separate until a clean unification is designed. Add a note in both files about
the duplication.

==> agreed, keep both then unify later.

**5. Splinification**

There are currently two `splinify` paths:

| | `Models/splinify` (ACEpotentials) | `et_models/splinify` (ACEpotentials) |
|-|-----------------------------------|--------------------------------------|
| Input | `LearnableRnlrzzBasis` + `ps.Wnlq` | `ETACE`/`ETPairModel` + params |
| Output | `SplineRnlrzzBasis` (Interpolations) | `TransSelSplines` (P4ML CubicSplines + KA) |
| Calling convention | `evaluate(spl, r, Zi, Zj, ps, st)` | `(l, X::AbstractVector{<:XState}, st)` |
| GPU-ready | no | yes |

They cannot be fully merged right now because their external interfaces differ:
`(r::Real, Zi, Zj)` vs batched `XState`. The `Models.ACEModel` calculator calls
the scalar interface; the `ETACE` calculator calls the batched `XState` interface.
Changing either would be breaking.

However, the **internal storage can be unified**: `SplineRnlBasis` will store its
data as P4ML `CubicSplines` objects instead of `Interpolations` objects. Concretely:

- `SplineRnlBasis` holds an `NZ×NZ` `SMatrix` of `CubicSplines{NX, LEN, T}`.
- `splinify(::LearnableRnlBasis, ps)` calls `P4ML.splinify(x -> Wnlq_ij * polys(x),
  -1.0, 1.0, nnodes)` for each `(iz0, iz1)` pair instead of
  `Interpolations.cubic_spline_interpolation`.
- Evaluation uses `P4ML._eval_cubspl` directly (same Hermite cubic math as
  `TransSelSplines`).

Benefits of this convergence:
1. **Drops `Interpolations` and `OffsetArrays`** from ET's new dependencies entirely
   (P4ML already uses Interpolations internally to get C2,2 regularity, so we get
   the same quality without a new ET dep).
2. The internal data layout `SVector{NX, SVector{LEN, T}}` is identical to what
   `TransSelSplines` uses, making the future GPU migration trivial: a
   `to_TransSelSplines(spl::SplineRnlBasis, selector)` converter just needs to pack
   the `NZ²` splines into the `TransSelSplines` structure.
3. Eliminates the `SPL_OF_SVEC` type alias (a complex `Interpolations` chain type)
   from the type signature of `SplineRnlBasis`.

The `et_models/splinify.jl` in ACEpotentials stays as-is for now: it operates on the
`EmbedDP`+`SelectLinL` architecture and produces `TransSelSplines` directly.

==> agreed: converge on P4ML CubicSplines internally, keep separate types for now

**6. Constructor**

`ace_learnable_Rnlrzz` becomes `learnable_Rnl_basis` (or keep the old name as a
re-export alias in ACEpotentials). It handles:
- polynomial type dispatch (`:legendre`, `:chebyshev`, `(:jacobi, α, β)`)
- envelope shorthand (`:poly2sx`, `:poly1sr`, `:x`, `:r`, `:r_ace1`)
- transform shorthand (`:agnesi`)
- default `maxq` from `maxn`

==> agreed to learnable_Rnl_basis, hypers are fine

**7. Dependencies**

The submodule needs:
- `Polynomials4ML` (already in ET) — also used internally for spline construction
- `StaticArrays` (already in ET transitively)
- `LuxCore` (already in ET)
- `ForwardDiff` (already in ET)
- `ChainRulesCore` (already in ET)

**No new dependencies required.** By using `P4ML.splinify` and `P4ML.CubicSplines`
internally (see §5 above), `Interpolations` and `OffsetArrays` are no longer needed
as direct ET dependencies. P4ML uses them internally to produce C2,2 splines, but
ET does not need to import them. 

**8. GPU / KernelAbstractions**

The `LearnableRnlBasis` evaluation path (matrix-vector multiply + ForwardDiff)
is CPU-only today, same as in ACEpotentials. No GPU path is introduced. The
splined variant via `TransSelSplines` (which lives separately in ET already) *is*
GPU-ready. This is the right division of labour for the first pass.

==> ok to do GPU as a second stage 

---

## Adapting ACEpotentials

After the ET submodule is in place, ACEpotentials changes are minimal:

1. **Remove** `src/models/radial_envelopes.jl`, `radial_transforms.jl`,
   `Rnl_basis.jl`, `Rnl_learnable.jl`, `Rnl_splines.jl`.
2. **Replace** with re-exports:
   ```julia
   import EquivariantTensors.Radials: LearnableRnlBasis, SplineRnlBasis,
                                       splinify, learnable_Rnl_basis
   # backward-compat aliases
   const LearnableRnlrzzBasis = LearnableRnlBasis
   const SplineRnlrzzBasis    = SplineRnlBasis
   ```
3. **Keep** `ace_heuristics.jl` thin wrapper that calls `learnable_Rnl_basis` from
   ET. The `ace_model` constructor changes only the import path.
4. **Keep** `et_models/splinify.jl` as-is for the GPU path.
5. Update `Project.toml` in ACEpotentials to tighten the ET version lower bound to
   the release that adds `Radials`.

==> keep these notes, but we do this later

---

## Additional functionality to consider for ET

The following are not yet in either package or are scattered in ways that would
benefit from consolidation in ET:

### Radial envelopes
- **Learnable envelope parameters**: allow `p` in `PolyEnvelope2sX` to be a
  Lux parameter rather than a fixed integer. This would let the model adapt the
  cutoff shape during training.
- **Soft-envelope from distance**: an envelope that is a function of `r` directly
  (not `x`) and uses a smooth cutoff like `(cos(π r/rcut) + 1)/2`. Already
  implicit in ACE1 but not available as a standalone component.

### Transforms
- **Unified scalar/edge transform**: a clean abstraction that can be called both
  as `t(r, iz, jz)` (scalar, for `LearnableRnlBasis`) and as `t(edge::XState)`
  (for `EmbedDP`), sharing the same parameter storage. This avoids the current
  duplication between `transforms/agnesi.jl` and the `radial_transforms.jl`
  being ported.
- **Learnable transform parameters**: expose `r0`, `a`, `p`, `q` as Lux
  parameters with appropriate initialization and regularization support.

### Basis / spec utilities
- **`sparse_AA_spec` and level functions** (`TotalDegree`, `SparseLevelSpec`, …)
  currently live in ACEpotentials. These are model-architecture-level decisions
  that belong in ET, not in a potentials package.
- **Orthonormal basis construction**: already added in ET (PR #105); the radial
  module should expose a `orthonormal_Rnl_basis` constructor that calls the
  existing orthonormalization code after the learnable basis is set up.

### Smoothness priors
- `algebraic_smoothness_prior`, `gaussian_smoothness_prior`, `exp_smoothness_prior`
  in ACEpotentials compute regularization matrices from the `LearnableRnlBasis`
  spec. These should move to ET once the basis type lives there.

### Pair potential radial basis
- The pair potential basis in ACEpotentials is constructed with the same
  `ace_learnable_Rnlrzz` machinery but restricted to `l = 0`. ET could expose a
  convenience constructor `pair_Rnl_basis(; ...)` that handles this common case.

==> all good points, keep these as notes for future work 

---

## Sequencing

Stage 1 (this PR — ET only):

1. Create `src/radials/` with the six files listed above; port code with the
   following changes from the ACEpotentials originals:
   - rename types (`LearnableRnlrzzBasis` → `LearnableRnlBasis`, etc.)
   - move `_i2z`/`_z2i`/`_get_nz` helpers to ET utils
   - replace `Interpolations.cubic_spline_interpolation` with `P4ML.splinify` +
     `P4ML.CubicSplines` in `SplineRnlBasis` and `splinify.jl`
   - no change to the `evaluate(basis, r, Zi, Zj, ps, st)` interface
2. Export from the main `EquivariantTensors.jl` module.
3. Add tests: construct a `LearnableRnlBasis`, call `splinify`, check that
   `evaluate` and `evaluate_ed` give matching values between the learnable and
   splined variants (numerical agreement to spline tolerance).
4. Tag a minor version of ET.

Stage 2 (ACEpotentials adaptation — separate PR, later):

5. Remove the five source files from `src/models/`.
6. Add re-exports and backward-compat aliases (see "Adapting ACEpotentials" above).
7. Update `Project.toml` lower bound on ET.
8. Run full ACEpotentials test suite.

Stage 3 (future):

9. GPU path: add `to_TransSelSplines(spl::SplineRnlBasis, selector)` converter so
   ET-based models can migrate to the KA-backed spline path.
10. Unified scalar/edge transform abstraction.
11. Move level/spec utilities and smoothness priors from ACEpotentials to ET.