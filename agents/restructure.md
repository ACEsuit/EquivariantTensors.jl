# ET Restructuring — Working Notes

Status: rev 2 (2026-06-11), CO's comments on rev 1 folded in as decision
records. Based on CO's initial thoughts, the research notes in
`projects/equivarianttensors/notes/` (background, eqcp, eqtucker), and a
survey of the current code.

---

## 1. Scope statement (proposed)

ET = parameterisation and evaluation of tensors `T` that are equivariant
under a group `G` (default O(3)), contracted against an equivariant
system embedding `A`:

```
F(X) = T : A(X)^⊗N,    A_nlm = Σ_j φ_nlm(x_j),    φ(gx) = ρ(g)φ(x)
```

The research notes establish the *forced architecture* (Schur +
connectedness): every manifestly equivariant finite-rank format is

```
T = Σ_{l,τ} C^τ_{l1...lN} ⊗ c^{l,τ}        (fixed CG carrier ⊗ free coeffs)
```

with equivariance living entirely in the fixed carrier `C` and all
compression living entirely in the G-trivial coefficient tensor `c`.
**Consequence for the package design: a "tensor format" in ET is a choice
of compression format for `c` plus a contraction strategy against the
pooled features. The carrier machinery (CG coupling, symmetrisation) is
shared infrastructure across all formats.**

**Generality of the group (CO correction, checked).** Compactness is not
needed; but "locally compact" is not the right hypothesis either — local
compactness gives a Haar measure but not complete reducibility (e.g. ℝ
has the non-semisimple rep `t ↦ [[1,t],[0,1]]`; same problem for the
Euclidean group). What the forced-architecture arguments actually use:

1. ρ finite-dimensional and **completely reducible (= semisimple)** —
   this gives the isotypic decomposition, the CG carrier, and Schur. It
   is a property of the *representation*, not the group, and holds
   automatically for (a) all continuous f.d. reps of *compact* groups
   (unitarisable via averaging), and (b) all f.d. reps of *connected
   semisimple* Lie groups (Weyl's complete reducibility theorem) — which
   covers the Lorentz group SO⁺(1,3) via its cover SL(2,ℂ). More
   generally, reductive groups whose centre acts semisimply.
2. Connected identity component G⁰, finite component group G/G⁰
   (O(1,3): ℤ₂×ℤ₂ — fine). The character argument survives
   noncompactness: χ: G → ℝ*, χ^N = 1 still forces χ(g) ∈ {±1}.

Suggested phrasing: *"finite-dimensional semisimple representation of a
Lie group with finitely many connected components"*, with compact groups
and connected semisimple groups (Lorentz) as the two guaranteed classes.

Caveats in the noncompact case, relevant to code design:
- f.d. reps of noncompact simple groups are never unitary; carrier
  "orthonormality" is then w.r.t. an invariant *bilinear form*, not a
  Hermitian inner product. Conditioning of CG bases may differ.
- Anything that *integrates or samples over G* — `quad_O3.jl`-style
  quadrature symmetrisation, test verification via random rotations —
  is compact-only. The CG path generalises; the quadrature path does
  not. The `groups/` layer should keep these two routes separate.

---

## 2. Mapping current code onto the pipeline

| Pipeline stage | Current code | Restructure verdict |
|---|---|---|
| particle embedding φ | `transforms/`, `embed/`, `lib/ACEradials`, P4ML, SpheriCart | out of core (lib / upstream), §5 |
| graph datastructure | `embed/graph.jl` | **stays in ET** (decision, §5) |
| particle states / DP diff | `transforms/diffnt.jl`, `embed/embeddings.jl`, `extensions/atoms.jl` | out of core, §5 |
| pooling → A | `ace/sparseprodpool*.jl` | keep in ET (decision, §4) |
| products A^⊗N | `ace/sparsesymmprod*.jl`, `symmprod_dag*.jl`, `static_prod.jl` | part of the *sparse format* |
| carrier (CG, symmetrisation) | `O3/`, `utils/symmop.jl` | shared core, promote |
| assembled format | `sparse_ace_basis/layer/ka/utils.jl` | becomes `formats/sparse/` |
| spec utilities | `utils/` (setproduct, invmap, sparseprod, selector) | shared core |
| species-select linear | `utils/selectlinl.jl` | chemistry-adjacent — lib? |

Note the current `src/ace/` mixes two distinct things: the generic pooling
(A) and the sparse-format-specific symmetric products (AA). They should
separate.

---

## 3. Proposed layout

```
src/
  groups/        # O3: irreps, CG, coupling trees, carrier/symmetrisation
                 # (abs. O3/, symmop.jl); flag generality, don't abstract yet
  graphs/        # ETGraph: system <-> edge-list 3-tensor/2-tensor reshaping
  pooling/       # PooledSparseProduct + KA kernels: embeddings -> A
  formats/
    sparse/      # current: sparsesymmprod + DAG + A2Bmaps
                 # dense = special case via a dedicated constructor (see §6)
    cp/          # (new) TRACE: symmetric CP of c (+ Schur channel mixing W)
    tucker/      # (new) per eqtucker.qmd (notes still to be written)
    tt/          # (new) tensor train — see §6, Hodapp/Shapeev prior art
  specs/         # spec generation/indexing utilities
lib/
  ACEradials/    # done
  (candidate) DP-coupled embedding machinery   # see §5
```

**Decision (CO): no `dense/` format.** Dense is a special case of sparse;
within the current storage/access structure a separate format adds
nothing — provide a convenience *constructor* of the sparse format
instead. Revisit only if a genuinely dense storage layout (BLAS-able
contractions instead of sparse indexing) turns out to matter.

Format interface — **decisions (CO):**
- 100% Lux compatibility is a hard requirement for all formats.
- No abstract supertype for now. An abstract type is only justified once
  there is shared functionality needing dispatch; start with the
  duck-typed I/O contract (A in; equivariant features per L out; specs /
  metadata; Lux layer semantics). (The existing `AbstractETLayer` only
  provides evaluate-allocation conveniences and should not silently grow
  into a format supertype.)
- Open design item: is the bespoke `evaluate`/`pullback`/`whatalloc`
  interface worth keeping *as public API* on top of standard AD? Likely
  resolution: keep the hand-written in-place kernels (they are the
  performance path, esp. with Bumper/KA) but expose them to the outside
  world only through ChainRules `rrule`s, so users interact via Lux +
  AD and never call `pullback!` directly. To be validated on the sparse
  format during step 2 of §8.

---

## 4. Pooling stays in ET  *(decided)*

Reasons (unchanged from rev 1): application-agnostic, small, and the
transformation law of A is what defines the constraint on T —
Aspec/AAspec/coupling co-evolve. Isolate as `src/pooling/` with a clean
boundary so graduating later stays cheap. ET's contract: *"give me
per-particle embeddings as plain arrays; I pool and contract."*

**New design question (CO): storage layout of A.** Flat vector with spec
indexing (current), or structured `(n, (l,m))` channels? Considerations:

- All new formats want the Schur block structure explicitly: the channel
  mixing W acts on n only, carrier contractions act on (l,m) only.
  Natural layout: per-l blocks `A^l ∈ ℝ^{n_l × (2l+1)}` (ragged
  vector-of-matrices), which is also what GPU-batched carrier
  contractions and Tucker/TT sweeps want.
- The sparse format is indifferent: it consumes A through an index spec
  and can address into per-l blocks as easily as into a flat vector.
- The flat vector is maximally flexible for *irregular* specs (different
  n-range per l, which we do use: `n_l` above).
- Middle ground: keep flat contiguous storage, add a lightweight block
  view (l ↦ matrix view) on top; formats pick their access pattern.

Verdict: probably *the* key data-structure decision of the restructure;
prototype both access patterns against the CP format before committing.

---

## 5. PState / DecoratedParticles boundary  *(decided, destinations refined)*

Survey result (rev 1): the DP/XState machinery lives entirely *upstream*
of A; the tensor core consumes and differentiates plain arrays.
**Decision: adopt the boundary** — ET core takes embedding arrays and
defines pullbacks w.r.t. those arrays only.

Destination recommendations (per-piece, refined after CO's comments):

- **Graph (`embed/graph.jl`): stays in ET** (CO). All models take graphs
  as inputs, and pooling relies on the ETGraph structure to convert the
  edge-embedding 3-tensor to the 2-tensor pooling layout and back. Only
  move out if a replacement plan exists. Refinement: keep `ETGraph`
  *container-agnostic* in `edge_data` so that ET does not need DP for the
  graph itself — DP enters only through what users store in the graph.
- **`diffnt.jl` (NamedTuple/Dual differentiation tooling): move into DP
  itself.** It is generic make-structs-differentiable tooling with no ET
  content; DP is its natural owner, and other DP consumers benefit.
- **`EmbedDP`, `decpart.jl`, `atoms.jl` extension: lib package** (e.g.
  `lib/ETAtoms` or `lib/ETEmbeddings`), not ACEpotentials — keeps the
  radials precedent (co-evolution in one repo, graduation path open) and
  keeps ACEpotentials a pure consumer.
- **DP as a `lib/` of ET: recommend no.** The `lib/` slot is for packages
  that *depend on* ET (radials pattern). After this restructure ET core
  no longer depends on DP at all, and DP is independently useful — it
  should remain a standalone package; the *coupling layer* (EmbedDP etc.)
  is what belongs in `lib/`.

Net effect: DecoratedParticles (and possibly Lux→LuxCore) drop out of
ET's hard deps; ETGraph stays but becomes representation-agnostic.

---

## 6. New formats — implementation notes

Per the eqcp analysis, all formats share Stage 1 (carrier) and differ in
Stage 2/3 (compression of c, evaluation strategy):

- **dense (= sparse constructor)**: c stored in full per (l,τ) block,
  realised as a constructor producing an un-pruned sparse format.
  Still useful early as the reference/testing ground for carrier reuse.
- **CP / TRACE**: Schur-admissible channel mixing
  `Ā_klm = Σ_n W_{kn} A_nlm` (W independent of l,m — `selectlinl`-like
  machinery may partially be reusable), then symmetric CP on the
  G-trivial coefficient tensor. Evaluation never forms c: per rank k,
  contract carrier against rank-1 channel products.
- **Tucker**: eqtucker.qmd is still a stub — code design should wait for /
  proceed jointly with those notes. Expected shape: shared factor matrix
  on the multiplicity (n) modes only (Schur forces factors to act
  trivially on (l,m)), Tucker core in the coupled basis.
- **TT**: the structural observation (the CG coupling tree is itself a
  sequential binary contraction, so the carrier is already a TT/HT
  network and a TT format for c can fuse with it) **is published prior
  art**: M. Hodapp & A. Shapeev, *Equivariant Tensor Network Potentials*,
  [arXiv:2304.08226](https://arxiv.org/abs/2304.08226), MLST (2024) —
  SO(3)-invariant-under-contraction tensor networks (MPS-like) for
  MLIPs, with follow-up work (e.g. dispersion-corrected ETN, JCP 2025).
  Consequence: the ET TT format is an engineering task with ETN as the
  reference; any *research* novelty must be relative to that paper
  (possible angles: O(3) incl. parity, alignment with the symmetrised /
  recoupled carrier, symmetric-TT for c, generic-G formulation) — to be
  assessed in the notes repo, not here.
- **others**: tensor ring, hierarchical Tucker (HT = general coupling
  trees rather than the sequential ACE tree). Park for later.

Shared open issue for all new formats: the S_N-symmetrisation /
recoupling bookkeeping (background.qmd, "where it complicates the CG
basis") — currently handled inside `symmetrisation_matrix` for the sparse
format; needs to be exposed as reusable carrier infrastructure.

---

## 7. Symmetric vs general tensors  *(decided)*

Symmetry is free for `F = T : A^⊗N`; non-symmetric T only matters when
slots carry different embeddings. **Decision: keep mode-symmetry a
property of the concrete format implementations; don't bake "all slots
identical" into the interface or carrier code (already
slot-heterogeneous). No general formats now.**

---

## 8. Sequencing sketch

1. Boundary cleanup first (no new functionality): split `src/ace/` into
   `pooling/` + `formats/sparse/`; promote O3+symmop to `groups/`;
   `graph.jl` → `src/graphs/` (made container-agnostic); move diffnt
   toward DP and EmbedDP/decpart/atoms toward a lib package.
2. Settle the format I/O contract on the sparse format (incl. the
   Lux/ChainRules-vs-bespoke-pullback question, §3) and the A storage
   layout question (§4). Behaviour-preserving; existing tests must pass.
3. Dense-via-sparse constructor as carrier-reuse validation.
4. CP/TRACE format.
5. Tucker (gated on eqtucker.qmd notes), then TT (after the
   novelty-vs-ETN assessment, §6).

All new format kernels: KA from day one (CO).

---

## 9. Decision record (from CO review, 2026-06-11)

- Group scope: state as "f.d. semisimple rep of a Lie group"; keep code
  O3-only but flag and avoid hard-coding where generality is cheap
  (e.g. pass irrep dimensions from `groups/` instead of inlining 2l+1
  in format code). Same object pattern (D-matrices + CG analogues) is
  expected to carry over to other groups.
- Real SH basis committed throughout the new formats; the carrier-level
  `basis = real/complex` switch stays in `groups/` so complex can be
  reintroduced later (no serious obstacle identified).
- No abstract format supertype initially.
- 100% Lux compatibility; KA from day one.
- Dense is a constructor of sparse, not a format.
- Species/categorical channels are part of the multiplicity index n —
  n is "the channel capturing everything invariant". (Lumping (l,n)
  would also be technically possible but goes against the grain of the
  literature — don't.)
- Graph stays in ET; pooling stays in ET.

## 10. Remaining open questions

- A storage layout: flat + spec indexing vs per-l blocks vs flat with
  block views (§4). Prototype against CP before committing.
- Public differentiation API: ChainRules-only surface over in-place
  kernels, or keep `evaluate`/`pullback` exported (§3).
- Naming: package-level vocabulary (carrier / coefficients / format?)
  and concrete type names for the new formats.
- DP follow-through: agree `diffnt` upstreaming with DP owners; pick the
  lib package name for the EmbedDP/atoms machinery (§5).
