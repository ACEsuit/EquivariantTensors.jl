# ET Restructuring — Working Notes

Status: rev 3 (2026-06-14). The **boundary cleanup is complete** (PRs
#110–#122; resulting layout in `docs/src/architecture.md`, PR history in
umbrella #109), so the completed sections below are compressed to "DONE"
stubs. What remains live is the **new-format work** (§3 interface, §4 A
storage, §6 CP/Tucker/TT + W-fold, §8) and the open questions in §10. The
design decisions (§1, §6.1, §7, §9) are kept as records. Based on CO's
thoughts, the research notes in `projects/equivarianttensors/notes/`
(background, eqcp, eqtucker), and a survey of the code.

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

## 2. Mapping current code onto the pipeline — DONE

The boundary cleanup that this section planned has landed: `src/ace/` split
into `pooling/` (A) + `formats/sparse/` (AA), `O3/`+`symmop` promoted to
`groups/`, and the particle/chemistry machinery moved out (see §5). The
**resulting layout is documented in the Architecture docs**
(`docs/src/architecture.md`); the **PR-by-PR history** is the checklist in
umbrella PR #109.

Still open from this section: the §1 caveat — separate the CG route from the
compact-only quadrature route inside `groups/` (quad_O3 stays in the O3 module
for now). Tracked in §10.

---

## 3. Proposed layout

The *realised* layout (groups/, graphs/, pooling/, formats/sparse/, utils/,
embed/, transforms/, the extensions and lib/ACEradials) is now documented in
`docs/src/architecture.md`. What remains live here are the **planned format
additions** (`formats/cp|tucker|tt`) and the **format-interface decisions**
below.

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

## 5. PState / DecoratedParticles boundary — DONE

**Decision adopted and fully landed.** ET core takes embedding arrays and
defines pullbacks w.r.t. those arrays only; it is now `LuxCore`-only with
DecoratedParticles wired in through `DecoratedParticlesExt`. The graph stays
in ET (container-agnostic `ETGraph`); chemistry/radials moved to
`lib/ACEradials`; `diffnt` moved into DP (v0.1.4); the Lux→LuxCore trim is
done (#121). **Live contract worth keeping:** particle/edge data must be
*state types* with tangent arithmetic (XStates, or arrays of `SVector`s) —
bare `NamedTuple`s are not supported as particles (CO, #110).

Boundaries are summarised in `docs/src/architecture.md`; PR-by-PR history is
in umbrella #109 (#110, #113–#117, #121).

Still open (deferred to a future `graphs/` step): make the edge container /
`NeighbourListsExt` fully container-agnostic (it currently has a DP trigger
and constructs PStates). Tracked in §10.

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

1. Boundary cleanup first (no new functionality) — **DONE**: `src/ace/` split,
   O3+symmop → `groups/`, `graph.jl` → `graphs/`, particle/chemistry machinery
   out of core, and the Lux→LuxCore trim (#121). See `docs/src/architecture.md`
   for the result and umbrella #109 for the PR history.
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

- ACEradials ↔ Polynomials4ML spline ownership — **DONE**: P4ML now owns the
  GPU-safe cubic-spline kernel and ACEradials' `TransSelSplines` calls it
  (Agnesi merge #119, P4ML helpers #122 → P4ML v0.5.10, de-fork #120). See
  `agents/radials.md`.

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
- Separate the CG route from the compact-only quadrature route inside
  `groups/` (quad_O3 currently stays in the O3 module; §1 caveat).
- Make the edge container / `NeighbourListsExt` fully container-agnostic
  (deferred to a future `graphs/` step; currently DP-triggered, §5).
