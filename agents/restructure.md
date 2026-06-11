# ET Restructuring — Working Notes

Status: draft for discussion (2026-06-11). Based on CO's initial thoughts,
the research notes in `projects/equivarianttensors/notes/` (background,
eqcp, eqtucker), and a survey of the current code.

---

## 1. Scope statement (proposed)

ET = parameterisation and evaluation of tensors `T` that are equivariant
under a compact group `G` (default O(3)), contracted against an
equivariant system embedding `A`:

```
F(X) = T : A(X)^⊗N,    A_nlm = Σ_j φ_nlm(x_j),    φ(gx) = ρ(g)φ(x)
```

The research notes establish the *forced architecture* (Schur + connectedness):
every manifestly equivariant finite-rank format is

```
T = Σ_{l,τ} C^τ_{l1...lN} ⊗ c^{l,τ}        (fixed CG carrier ⊗ free coeffs)
```

with equivariance living entirely in the fixed carrier `C` and all
compression living entirely in the G-trivial coefficient tensor `c`.
**Consequence for the package design: a "tensor format" in ET is a choice
of compression format for `c` plus a contraction strategy against the
pooled features. The carrier machinery (CG coupling, symmetrisation) is
shared infrastructure across all formats.** This is the single most
important structural insight from the notes and should drive the layout.

==> small correction, the group need not be compact, all that is needed is a 
    finite-dimensional representation. So this applies in particular also 
    to the lorentz group after embedding into a finite-dimensional space
    not sure what the correct terminology is, but may be  
    "finite-dimensional representation of a locally compact Lie group" ??? 
    Please think about it and check. 

---

## 2. Mapping current code onto the pipeline

| Pipeline stage | Current code | Restructure verdict |
|---|---|---|
| particle embedding φ | `transforms/`, `embed/`, `lib/ACEradials`, P4ML, SpheriCart | out of core (lib / upstream) |
| graph / particle states | `embed/graph.jl`, `extensions/atoms.jl`, DP dep | out of core (see §5) |
| pooling → A | `ace/sparseprodpool*.jl` | keep in ET (see §4) |
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
  groups/        # G-interface + O3 default: irreps, CG, coupling trees,
                 # carrier/symmetrisation construction (abs. O3/, symmop.jl)
  pooling/       # PooledSparseProduct + KA kernels: embeddings -> A
  formats/
    sparse/      # current: sparsesymmprod + DAG + A2Bmaps
    dense/       # (new) unconstrained c on the B-basis
    cp/          # (new) TRACE: symmetric CP of c (+ Schur channel mixing W)
    tucker/      # (new) per eqtucker.qmd (notes still to be written)
    tt/          # (new) tensor train — see §6
  specs/         # spec generation/indexing utilities
lib/
  ACEradials/    # done
  (candidate) particle-state / graph / atoms machinery   # see §5
```

==> dense is a special case of sparse. And I'm not convinced there is much 
    value in having this as a separate format rather than a special 
    constructor. At least this is how I see it within the current code 
    structure. If the storage and access formats for the tensors changes 
    significantly then this could be wrong. 

Common format interface (all Lux layers, KA-compatible kernels):
`evaluate(fmt, A) -> (B_L for L in LL)`, plus `pullback`, `whatalloc`.
Formats differ *structurally* in how they contract — sparse goes through
explicit AA products; CP goes through channel compression then per-rank
products; dense/Tucker/TT through their own contractions — so the shared
abstraction should be the I/O contract (A in, equivariant features per L
out, specs/metadata), not the internal evaluation path.

==> Ensure 100% lux compatibility. Consider whether a separate 
    `evaluate` and `pullback` interface is worthwhile on top of the 
    standard AD (possibly with ChainRules)

---

## 4. Design question: does pooling belong in ET?

Recommendation: **keep it in ET core.** Reasons:
- It is fully application-agnostic (no chemistry; just fused sparse
  product + sum over particles) — unlike radials, it carries no domain
  conventions, so the ACEradials argument for eviction does not apply.
- The transformation law of A is what *defines* the constraint on T;
  Aspec/AAspec/coupling are co-designed and co-evolve. Splitting them
  across packages recreates the two-repo coordination cost that
  `agents/radials.md` cites as the reason for `lib/`.
- It is small (one struct + kernels).

But: isolate it as its own top-level concern (`src/pooling/`) with a clean
boundary, so graduating it later stays cheap. ET's contract becomes:
*"give me per-particle embeddings as plain arrays; I pool and contract."*


==> I like that, let's do this. One question to consider is whether storing 
    `A` as a single vector is a good thing, or should it be stores as a 
    tensor in `(n, (l, m))` channels? Unclear to me, and maybe an important
    design decision for readability and code efficiency?

---

## 5. Design question: PState / DecoratedParticles

Survey result: the DP/XState machinery (`transforms/diffnt.jl`,
`transforms/decpart.jl`, `embed/embeddings.jl`, `embed/graph.jl`,
`extensions/atoms.jl`) lives entirely *upstream* of A. The tensor core
(pooling, products, coupling, formats) never sees an XState — it consumes
and differentiates plain arrays.

Recommendation: **make this an explicit boundary.** ET core takes
embedding arrays (e.g. Rnl, Ylm matrices, or a general Φ) and defines
pullbacks w.r.t. those arrays only. The particle-state representation and
differentiation-through-structs question then becomes a *consumer-side*
decision (ACEpotentials or a lib package, e.g. `lib/ETGraphs` or similar),
and revisiting the PState design choice no longer blocks or entangles the
ET restructure. I.e. don't answer "was PState a good idea" inside ET —
move the question out of scope. The Dual-number tricks in `diffnt.jl` are
self-contained and would move wholesale.

This would also drop DecoratedParticles (and possibly Lux vs LuxCore)
from ET's hard deps.

==> Ok, let's do that. Should the differentiation tooling around 
    DecoratedParticles (=DP) be moved into DP? And would it make sense 
    to make DP a lib for ET? 
  

---

## 6. New formats — implementation notes

Per the eqcp analysis, all formats share Stage 1 (carrier) and differ in
Stage 2/3 (compression of c, evaluation strategy):

- **dense**: c stored in full per (l,τ) block. Cheapest to implement;
  useful as reference/testing ground for the others; essentially the
  current linear ACE with dense instead of pruned-sparse c.
- **CP / TRACE**: Schur-admissible channel mixing
  `Ā_klm = Σ_n W_{kn} A_nlm` (W independent of l,m — `selectlinl`-like
  machinery may partially be reusable), then symmetric CP on the
  G-trivial coefficient tensor. Evaluation never forms c: per rank k,
  contract carrier against rank-1 channel products.
- **Tucker**: eqtucker.qmd is still a stub — code design should wait for /
  proceed jointly with those notes. Expected shape: shared factor matrix
  on the multiplicity (n) modes only (Schur forces factors to act
  trivially on (l,m)), Tucker core in the coupled basis.
- **TT**: observation worth recording — the CG coupling tree is itself a
  sequential binary contraction, i.e. **the carrier is already a TT/HT
  network** (cores indexed by intermediate L_2,...,L_{N-1}). A TT format
  for c whose ranks align with the coupling-tree structure could fuse
  carrier and coefficient contraction into one sweep. This may be the
  most natural format of all and possibly a research contribution in
  itself.
- **others**: tensor ring, hierarchical Tucker (HT = general coupling
  trees rather than the sequential ACE tree). Park for later.

Shared open issue for all new formats: the S_N-symmetrisation /
recoupling bookkeeping (background.qmd, "where it complicates the CG
basis") — currently handled inside `symmetrisation_matrix` for the sparse
format; needs to be exposed as reusable carrier infrastructure.

==> check whether your ideas about the TT format are already published 
    by Alex Shapeev and Max Hodapp? (about 2-3 years ago)

---

## 7. Design question: symmetric vs general tensors

The notes prove symmetry is *free* for the use case `F = T : A^⊗N`
(contraction against a tensor power sees only Sym(T)). Non-symmetric T
only matters when the N slots carry *different* embeddings (e.g.
A ⊗ A' mixed contractions, message passing layers, multiple interacting
densities).

Recommendation: keep mode-symmetry as a property of the concrete format
implementations, but do not bake "all slots identical" into the abstract
interface or the carrier code (the CG machinery is already
slot-heterogeneous: ll-tuples need not be constant). Don't implement
general formats now; leave the door open.

==> ok agreed. 

---

## 8. Sequencing sketch

1. Boundary cleanup first (no new functionality): split `src/ace/` into
   `pooling/` + `formats/sparse/`; promote O3+symmop to `groups/`;
   decide destination for embed/transforms/DP machinery and move it.
2. Introduce the abstract format interface; make the sparse format
   conform to it. (Behaviour-preserving; existing tests must pass.)
3. Add the dense format (small, validates the interface + carrier reuse).
4. CP/TRACE format.
5. Tucker (gated on eqtucker.qmd notes), then TT.

==> agreed, but let's first iterate on whether dense is a separate format or not.

---

## 9. Open questions for CO

- Name for the format abstraction (`AbstractEquivariantFormat`?
  `AbstractCoupledTensor`?) and for the package-level vocabulary
  (carrier / coefficient / format?).

==> It is not necessary to have an abstract supertype. This is only useful 
    if there is shared functionality that requires dispatch. So initially 
    just don't introduce this. 

- real vs complex SH basis: keep dual support in the carrier throughout,
  or commit to real in the new formats?

==> commit to real throughout, unless you see very serious difficulties 
    in re-introducing complex later. 

- Generalise `O3` to a `groups/` interface *now* (O(3) as the one
  instance) or keep O3-concrete and abstract only when a second group
  (O(2)? Euclidean? lattice point groups?) becomes concrete?

==> This is a very good question. My sense is that ultimately the same 
    objects will be used, some analogue of D matrices and some analogue 
    of cg coefficients. This suggests that we should focus on O3 only 
    but where there are concerns about generality, flag them and 
    avoid hard-coding too much? 

- Where do categorical/species channels live conceptually in the
  non-atomistic setting — part of the multiplicity index n (my reading of
  TRACE's W_{zn}), or a separate mechanism (`selectlinl`)?

==> yes exactly, lump it into n. I am thinking of n as a channel that 
    captures everything invariant. Technically one could even lump (l, n)
    but my sense is this would become confusing since it goes against the 
    grain of the literature. 

- Destination for the particle/graph/DP machinery: lib package vs
  ACEpotentials? (§5 argues only that it leaves ET core.)

==> Give me your recommendation for the particle and DP machinery (see my question above) 
==> For the graph, I am more sceptical about moving it out of ET. 
    Initially at least all models will take graphs as inputs. And I would 
    argue the pooling relies on that datastructure, because it specifies how 
    to convert a 3-tensor into a 2-tensor for pooling and then invert.
    I would only move this out of ET if there is a good plan in place to 
    replace this functionality. 

- GPU: all new formats KA-from-day-one, or CPU-first like radials?

==> KA from day one. 
