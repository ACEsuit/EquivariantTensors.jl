# Architecture

This page describes the structure of `EquivariantTensors.jl` (ET) and how it
sits within the surrounding ACEsuit ecosystem: which package owns what, and the
dependency boundaries that were drawn deliberately. It is an orientation map,
not an API reference — see the [Public API](api.md) and
[Docstrings](docstrings.md) pages for the callable interface.

## Ecosystem overview

```
        Polynomials4ML          (upstream: general-purpose ML basis library)
              │
              ▼
       EquivariantTensors        (this package: equivariant tensor formats)
              │
              ▼
          ACEradials             (downstream: ACE-specific radial bases)

   DecoratedParticles ─ ─▶ EquivariantTensors   (loaded only via an extension)
```

The arrows are hard dependencies and are acyclic. `DecoratedParticles` (DP) is
**not** a hard dependency of ET core; it is wired in through a package
extension that activates when a consumer also loads DP.

## ET core layout (`src/`)

ET is organised into submodule directories, each a stage of the pipeline
`embeddings → pool → A → contract → equivariant features`:

| Directory | Purpose |
|-----------|---------|
| `groups/` | O(3) representation theory: irreps, Clebsch–Gordan coupling, coupling trees, and carrier symmetrisation — the *carrier* of the equivariant tensor. Exposes the `O3` module. |
| `graphs/` | The `ETGraph` data structure (system ↔ edge-list reshaping), the `EdgeEmbed` per-edge embedding adapter, and the atomic-system entry points. Container-agnostic. |
| `transforms/` | Input transforms on particle states (`StateTransform`). |
| `embed/` | Embedding layers: `StateEmbed` (transform → basis → post). |
| `pooling/` | `PooledSparseProduct` and its KernelAbstractions kernels: per-edge embeddings → pooled features `A`. |
| `formats/sparse/` | The sparse tensor format: `SparseSymmProd`, the sparse ACE basis/layer, and A2B symmetrisation maps. The reference format. |
| `utils/` | Shared utilities: spec generation/indexing, `SelectLinL` (categorical linear layer), weight initialisers (`et_zeros`/`et_normal`), type promotion/`adapt`. |
| `testing/` | Test helpers (`rand_graph`, …). |

ET layers are Lux layers (`AbstractLuxLayer` from LuxCore); a few carry the
internal `AbstractETLayer` supertype, which only provides evaluate/allocation
conveniences and is deliberately *not* a format supertype.

## Extensions (`ext/`)

ET core depends only on `LuxCore`; heavier or domain-specific glue lives in
package extensions that auto-load with their trigger packages:

- **`DecoratedParticlesExt`** (trigger: `DecoratedParticles`) — all evaluation
  and differentiation methods for `StateTransform`/`StateEmbed` on XState particle
  data. Calling these without DP loaded fails loudly (acceptable: every real
  consumer loads DP).
- **`NeighbourListsExt`** (triggers: `NeighbourLists`, `AtomsBase`,
  `DecoratedParticles`) — builds an `ETGraph` from a neighbour list / atomic
  system (`Atoms.interaction_graph`, `Atoms.nlist2graph`).

## `lib/ACEradials`

`lib/ACEradials` is a **separately registered subdirectory package** that
depends on ET. It holds the chemistry-specific radial-basis assembly that is
deliberately kept out of the application-agnostic ET namespace: the learnable
and splined `Rnl` bases (`LearnableRnlBasis`, `SplineRnlBasis`, `splinify`),
scalar and species-pair Agnesi transforms, cutoff envelopes, and the GPU
spline radial basis `TransSelSplines`. Its `AtomsBaseExt` supplies
bond-length / cutoff defaults from `AtomsBase`.

## Related packages

- **[Polynomials4ML](https://github.com/ACEsuit/Polynomials4ML.jl)** (P4ML) —
  the upstream, general-purpose ML basis library: orthogonal polynomials,
  trigonometric and spherical harmonics, atomic orbitals, and cubic splines,
  all with a unified batched/GPU evaluation interface. ET uses it for the
  radial and angular bases; ACEradials additionally uses its cubic-spline
  kernel. P4ML deliberately depends only on `LuxCore` so it stays reusable
  outside ACE.
- **[DecoratedParticles](https://github.com/ACEsuit/DecoratedParticles.jl)**
  (DP) — the particle/edge *state* types (XStates) and their tangent
  arithmetic / differentiation tooling. Loaded into ET through the extension
  above rather than as a hard dependency.

## Deliberate boundaries

The dependency directions encode design decisions, not just convenience:

- **P4ML stays `LuxCore`-only.** Anything pushed upstream into P4ML must be a
  pure, low-dimensional numeric kernel — no DP / AtomsBase / Lux.
- **ET core is `LuxCore`-only.** DecoratedParticles enters through an
  extension; ET core consumes plain embedding arrays and defines pullbacks with
  respect to those arrays. (`Lux` proper is only a test dependency.)
- **ACEradials sits downstream** and owns chemistry semantics (atomic-number
  indexing, species-pair blocks, envelopes), so ET stays application-agnostic.
- **Particle/edge data must be state types** with tangent arithmetic (XStates,
  or arrays of `SVector`s) — bare `NamedTuple`s are not supported as particles.
