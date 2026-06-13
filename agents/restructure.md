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

*Done (PR `restruct_acesplit`):* `src/ace/` split into
`pooling/` (sparseprodpool + KA kernels) and `formats/sparse/`
(sparsesymmprod, symmprod_dag (dormant, not included), sparse_ace_*,
sparsemat_ka); `static_prod.jl` → `utils/` since its kernels are shared
by pooling and the sparse format. Test tree mirrors the split
(`test/pooling/`, `test/formats/sparse/`, `test/utils/`). Pure moves,
no code changes beyond include paths.

*Done (PR `restruct_groups`):* `src/O3/` → `src/groups/O3/` and
`utils/symmop.jl` → `groups/symmop.jl` (the carrier symmetrisation
belongs with the group layer, not generic utils). Group tests collected
in `test/groups/`. Pure moves. The §1 caveat (separate the CG route
from the compact-only quadrature route inside `groups/`) remains open —
quad_O3 stays inside the O3 module for now.

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
ext/
  (rev 3) DP-coupled embedding machinery as DP-triggered extension  # §5
lib/
  ACEradials/    # done; also receives bond_len/agnesi defaults (§5)
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
  *Done (PR `restruct_graphs`):* `embed/graph.jl` → `src/graphs/graph.jl`
  and `extensions/atoms.jl` → `src/graphs/atoms.jl` (keeping the `Atoms`
  module); the single-file `src/extensions/` directory retired.
  Container-agnostic was already satisfied (ETGraph is fully parametric
  and DP-free; the PState question was handled by the DP trigger in
  #110), so this was a pure move.
- **`diffnt.jl` (NamedTuple/Dual differentiation tooling): move into DP
  proper** (not a DP→ForwardDiff extension). It is generic
  make-structs-differentiable tooling with no ET content; DP already
  depends on NamedTupleTools + StaticArrays, so ForwardDiff is the only
  new (light) dep. Against the extension route: differentiability is
  core to DP's purpose, and extension glue triggered by a package that
  end users never load directly (ForwardDiff) silently fails to
  activate — the classic extension footgun.
- **`EmbedDP`, `decpart.jl`: Pkg extension of ET triggered by DP**
  (rev 3, supersedes the rev-2 lib recommendation). Rationale: this is
  a compat shim whose design is under reconsideration — registering a
  lib package for code that may be redesigned/deprecated has no payoff,
  while an extension achieves the dependency goal (no hard DP dep in
  core; glue auto-loads for ET+DP consumers like ACEpotentials).
  Implementation notes: (a) `embeddings.jl` imports
  `AbstractLuxWrapperLayer`/`ContainerLayer` via Lux but they live in
  LuxCore — switch the import, no Lux trigger needed; (b) `decpart`
  methods on plain NamedTuples (the `NTorDP` union) can stay in core,
  only XState methods go in the extension. Trade-off accepted:
  extension code rides ET's version train.
- **`extensions/atoms.jl` prototypes + ext/: keep, but split.** The
  graph half (`interaction_graph`, `nlist2graph`) is the system→ETGraph
  entry point and stays with `graphs/`; note `NeighbourListsExt`
  currently constructs PStates directly — must become container-agnostic
  (or gain a DP trigger) once DP leaves the hard deps.
  (`forces_from_edge_grads` was deleted in `restruct_edgegrads`; its
  generic core replacement is `node_grads_from_edge_grads`, see §10.) The chemistry half (`bond_len`, agnesi
  defaults in `AtomsBaseExt`) belongs with ACEradials (which already
  has `elements.jl`/`transforms.jl`), not ET. Weakdep stubs themselves
  are harmless — no load cost, a dozen empty functions.
*Done (PR `restruct_chemistry`):* chemistry/radials moved to
lib/ACEradials: `transforms/agnesi.jl` → `ACEradials/src/agnesi_dp.jl`
(its 6-arg constructor is now a method of `ACEradials.agnesi_transform`;
unification with the scalar `GeneralizedAgnesiTransform` still pending,
see agents/radials.md §4); ET's whole `AtomsBaseExt` (bond_len +
LENGTHSCALES data + agnesi defaults) → ACEradials AtomsBase-extension;
`bond_len`/`agnesi_transform` stubs removed from `ET.Atoms` (graph stubs
stay); ET drops the AtomsBase and (vestigial) AtomsBuilder weakdeps —
both remain test-only deps. `test_agnesi.jl` reactivated in ACEradials
(fixed on arrival: bare-NamedTuple inputs → PStates per the #110
decision, and stale `s0/s1` field names → the live `z0/z1` convention).

- **DP as a `lib/` of ET: no.** The `lib/` slot is for packages that
  *depend on* ET (radials pattern). After this restructure ET core no
  longer depends on DP at all, and DP is independently useful — it
  remains a standalone package.

Net effect: DecoratedParticles (and possibly Lux→LuxCore) drop out of
ET's hard deps; ETGraph stays but becomes representation-agnostic.

### 5.1 DP removal — implementation record (PR `restruct_rmdp`)

Status: diffnt landed in DP v0.1.4 (registered); ET side done. Deviations
and refinements relative to the plan above:

- **NamedTuples are no longer supported as particle/edge types**
  (decision, CO review on PR #110). The embedding layers and the
  reshape/pad machinery require *state* containers; the precise contract
  is "particle/edge data must support tangent arithmetic and a tangent
  zero" (`zero`, `+`, scalar `*`) — XStates qualify, plain arrays of
  SVectors qualify, bare NamedTuples don't. Evidence collected in this
  PR: (a) NT edge data broke pullbacks (`Float32 * NamedTuple`
  undefined); (b) "zero of a NamedTuple" is ill-defined for categorical
  fields — the old byte-zeroing `__zero` hack fabricated invalid values,
  and the PState/VState point-vs-tangent distinction is the actual
  answer; (c) supporting both containers duplicated every `DPTransform`
  method across core and ext. The NT differentiation tooling itself
  stays in DP (generic, useful); ET just doesn't route particles through
  bare NamedTuples. `ETGraph` remains storage-agnostic.
- **`EmbedDP`/`DPTransform` structs stay in core** — core files
  (agnesi, transsplines) dispatch on them — but *all* evaluation and
  differentiation methods live in the new `ext/DecoratedParticlesExt.jl`.
  Consequence: calling a `DPTransform`/`EmbedDP` throws MethodError
  unless DP is loaded (loud failure, acceptable: all real consumers
  load DP).
- The `__zero` helper is deleted; `reshape_embedding` /
  `rev_reshape_embedding` pad with `zero(eltype)`, which DP provides for
  XState types (incl. `_mod_zero` for categorical fields). Also fixed:
  `rev_reshape_embedding` previously used `zero` while its forward
  counterpart used the hack — now consistent.
- **`NeighbourListsExt` gains DP as a second trigger**
  (`["NeighbourLists", "DecoratedParticles"]`) — took the cheap option;
  making it container-agnostic is deferred to the graphs/ step.

*Trigger analysis (PR `restruct_edgegrads`).* After `restruct_chemistry`,
ET's only atoms extension is `NeighbourListsExt` = the
neighbourlist→ETGraph builder, and its trigger
`["NeighbourLists", "DecoratedParticles"]` is correct:
  - *NeighbourLists* must be a trigger — the ext calls
    `NeighbourLists.PairList`; an extension may use only its triggers +
    the parent's hard deps, and NeighbourLists is deliberately not an ET
    hard dep.
  - *AtomsBase* was not a trigger under NeighbourLists 0.5 —
    `AbstractSystem` and the accessors reached through
    `NeighbourLists.AtomsBase` (a hard dep + re-export of NL 0.5).
    **Superseded by PR `restruct_nlist06`:** NeighbourLists 0.6 dropped
    AtomsBase as a hard dep and stopped re-exporting it (the
    `PairList(sys, rcut)` constructor moved into NL's own
    `NeighbourListsAtomsBaseExt`). So under 0.6 AtomsBase **must** be a
    trigger — the ext now imports it directly (`ustrip` via AtomsBase's
    re-export), and the NL sys-constructor activates because AtomsBase
    (which pulls Unitful transitively) is loaded. Trigger is now
    `["NeighbourLists", "AtomsBase", "DecoratedParticles"]`; compat
    bumped to NeighbourLists 0.6 only.
  - *DecoratedParticles* is a trigger because the ext **constructs
    PStates** for all edge/node data (per #110, particles must be state
    types; bare NamedTuples lack tangent arithmetic). Revisit only when
    the graphs/ step parametrises the edge container.
  - Transitive loads count, so ACEpotentials-style consumers (hard deps
    on NeighbourLists + AtomsBase + DP) activate it automatically; direct
    ET script users need `using NeighbourLists, AtomsBase,
    DecoratedParticles`.
  - *Decision (CO, PR `restruct_atomsdoc`):* keep the builder as the
    NeighbourListsExt and **document** the load requirement (docstrings
    on `Atoms.interaction_graph`/`nlist2graph`). The 3-trigger set is
    intrinsic (NL engine + AtomsBase system + DP states) and the friction
    is REPL-only — downstream packages auto-activate. A one-import `lib/`
    glue package was considered and deferred until the graphs/ builder
    form settles (same reasoning as the EmbedDP lib-vs-ext call).
- **`TransSelSplines` signatures relaxed** from
  `AbstractVector{<: XState}` to `AbstractVector` (duck-typed; the
  tangent-arithmetic contract applies, not a container restriction).
- **`Testing.rand_graph` keeps PState default edge data**, provided
  through the extension (`_default_randedge` stub in core, method in
  the ext).
- `NamedTupleTools` dropped from deps (only diffnt used it).
- Lux→LuxCore *not* done here (DP-focused PR); separate step.

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

### 6.1 Where does W live? (ET vs ACEradials)

Observation (CO): the channel mixing `Ā_klm = Σ_n W_kn A_nlm` *is* the
Tucker factor matrix and the CP/TRACE channel weights — and since
pooling is linear in the embedding, `W·A = pool(W·R · Ylm)`, i.e. W is
equivalently a *learned radial basis*. This is the code-level shadow of
eqcp.qmd's gauge freedom #2 ("how the multiplicity compression W is
distributed relative to the CP step").

Resolution: **the operation lives in both places, with genuinely
different meanings**, connected by an explicit converter:

- **ET** owns the primitive: the *G-equivariant learnable linear
  layer*. By Schur, any equivariant linear map acts arbitrarily on the
  multiplicity index n and as identity on (l,m) — this block structure
  is the unique equivariant linear layer, it is representation theory
  (ET's domain), and the CP/Tucker formats are *defined* through it.
  Dependency direction forces this anyway (ACEradials depends on ET).
  Computationally the post-pooling placement is the good one: one
  dense matmul per l-block per pooled A (BLAS/KA-friendly), not one
  per edge.
- **ACEradials** keeps `Wnlq * P(x)` as a *basis parameterisation*
  device: chemistry semantics (per-species-pair blocks, envelopes,
  smoothness priors, orthonormalisation) and the splinification /
  deployment path. Different granularity too: ACEradials' W is per
  (z_i, z_j), ET's acts on the full multiplicity index n (which
  includes species, per §9).
- **Bridge = a "fold" converter in ACEradials** (it depends on ET and
  owns splinify): absorb a trained ET channel mixing into the radial
  basis (`W_fmt ∘ W_rad`, then splinify) for production. The
  pooling-linearity identity gets one documented home instead of
  being accidental duplication.

Two caveats worth recording:
- The fold equivalence holds *only for shared-factor (symmetric)
  formats*: per-slot Tucker factors W_t cannot be absorbed into a
  single radial basis (would need one pooled A per slot). The
  pre-pooling placement hard-codes the shared-factor assumption; the
  ET-side placement keeps per-slot freedom open — another reason ET's
  version is the more primitive object.
- Gauge redundancy: W_fmt and W_rad both learnable is a product of two
  linear maps — pure overparameterisation. Constructors/docs should
  steer toward one learnable side (e.g. fixed orthonormal radials +
  learnable format mixing for training; folded splined radials for
  deployment). ET should not enforce this.

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
   `graph.jl` → `src/graphs/` (done, PR `restruct_graphs`; also absorbed
   `extensions/atoms.jl` and retired the `extensions/` directory; the
   `Atoms` module name kept); move diffnt toward DP (done), EmbedDP/
   decpart into a DP-triggered extension (done), and the
   bond_len/agnesi-defaults chemistry toward ACEradials (done, §5).
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
- Particle/edge data must be state types with tangent arithmetic
  (XStates; plain SVectors also qualify) — bare NamedTuples are not
  supported as particles (2026-06-12, CO review on PR #110; see §5.1).

## 10. Remaining open questions

Test-suite cleanup (2026-06-12, PR `restruct_groups`): dormant /
superfluous tests parked in `test/dormant/` (see its README).
Outstanding items: review `test_lux_models.jl`
(only CPU-vs-GPU model consistency coverage); revisit `test_embed.jl`
(SelectLinL coverage vs test_splines overlap) before finalizing the
restructure.

- ACEpotentials forces/virial wrapper: `restruct_edgegrads` deleted
  `forces_from_edge_grads` and replaced it with the generic core
  primitive `node_grads_from_edge_grads(G, w_edges)` (position part of
  the adjoint of `(x, cell) ↦ 𝐫_e`, no force sign). The forces wrapper
  (apply the `−` sign, units, `AbstractSystem` handling) and the virial
  (the cell part of the same adjoint) belong in ACEpotentials.jl — to be
  added there (repo not in this workspace).
- **Revisit `node_grads_from_edge_grads` naming + specification.** It is
  nothing special: it is the standard *pullback of a node → edge
  operation* (gather `x ↦ 𝐫_e`), and should be designed/named as such —
  i.e. as one half of a `forward (node→edge) + pullback (edge→node)`
  pair on the graph, ideally with a ChainRules `rrule` so it composes
  with AD rather than being a bespoke helper. The current name and
  ad-hoc signature are placeholders; settle this together with the
  graphs/ step and the §3 differentiation-API decision.
- A storage layout: flat + spec indexing vs per-l blocks vs flat with
  block views (§4). Prototype against CP before committing.
- Public differentiation API: ChainRules-only surface over in-place
  kernels, or keep `evaluate`/`pullback` exported (§3).
- Naming: package-level vocabulary (carrier / coefficients / format?)
  and concrete type names for the new formats.
- DP follow-through: land `diffnt` in DP (adds ForwardDiff dep there);
  confirm the EmbedDP/decpart-as-extension recommendation (§5, rev 3).
