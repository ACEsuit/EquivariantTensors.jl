# CP / TRACE format — implementation plan

Status: rev 2 (2026-06-15, addressed CO review; rev 1: 2026-06-14). Design
record for the **equivariant CP / TRACE** tensor format in ET. Based on the
research notes
(`projects/equivarianttensors/notes/`: `eqcp.qmd`, `background.qmd`,
`eqtucker.qmd`), the TRACE paper (arXiv:2210.01705, PRL 131 028001) and the
MACE paper (arXiv:2206.07697, NeurIPS 2022), and a survey of the ET code
(`formats/sparse/`, `groups/O3/`, `groups/symmop.jl`, `pooling/`,
`utils/selectlinl.jl`). Complements `restructure.md` (drills into its §6
CP bullet) — cross-references are to that file unless stated otherwise.

This is a notes doc (prose + pseudocode + ASCII). It proposes; it does not
implement. Concrete, opinionated where it can be; genuine open decisions for
CO are flagged `[CO]`.

---

## 0. Implementation status (2026-06-16, branch `restruct_tracce`)

**A first, fully-validated CPU reference implementation has landed.** Files:

- `src/formats/cp/equiv_linear.jl` — `EquivLinearL`, the Stage-2 channel-mixing
  primitive (`Āᵏ_{lm} = Σ_n Wˡ_{kn} A_{nlm}`; per-l, identity on m). Standalone
  Lux layer + `rrule`. Tucker will reuse it (§5).
- `src/formats/cp/cp_ace_basis.jl` — `CPACEbasis` + `cp_equivariant_tensor(;
  LL, mb_spec, Rnl_spec, Ylm_spec, basis, rank)`. Owns `W` (delegated to the
  mixer). Builds the carrier by calling `sparse_equivariant_tensors` on a
  **single-channel** `mb_spec` (all `n=1`) and reusing its `aabasis`
  (`SparseSymmProd`) + `A2Bmaps` — the never-form-`c` per-rank evaluation
  (mix → symmetric product → carrier). Full hand-written `rrule` (∂A, ∂W).
- `src/formats/cp/cp_ace_layer.jl` — `CPACElayer`, the Stage-3b `λ` readout
  (reuses `_tupmul`, which already differentiates `Matrix{SVector}·Matrix`).
- `test/formats/cp/test_cp_basis.jl` — wired into `runtests.jl`.

**Key realisation that shaped the implementation.** After Stage-2 mixing each
`l` carries a *single* channel, so the symmetric product treats equal-`l`
factors as identical. The carrier for that is exactly `symmetrisation_matrix`
on a single-channel `mb_spec` (all `n=1`) — so the entire Stage-1/3a carrier is
reused verbatim from the sparse format with **zero new coupling code**. This is
not just convenient; it is the correct TRACE symmetry (§2). The multi-channel
richness is recovered by the rank-`K` sum, as it should be.

**Validation (all green, three independent checks):**
1. **Sparse oracle** — for a single radial channel with `K=1, W=1`, `CPACEbasis`
   is *bit-identical* (`err = 0`) to the established sparse B-basis.
2. **Equivariance** — `F_0` rotation-invariant, `F_1` covariant with the real
   Wigner-D, both to machine precision (~1e-15), on pooled `A` from random
   configs.
3. **FD gradients** — Zygote vs finite differences w.r.t. `A`, `W`, `λ`
   (~1e-10 / ~1e-5).

Full suite: 5290 pass / 1 broken (the +24 are CP).

**What this first cut deliberately does NOT do yet (next steps, by priority):**

1. **KA / GPU kernels.** The reference is plain-Julia CPU (batched over nodes:
   `A` is `nnodes × |Aspec|`). The `k`-loop, the per-`l` mix, the carrier
   multiply, and the `λ`-combine should each become KA kernels (§5 GPU story),
   with the `k`-axis a launch dimension. The pullbacks already separate
   `∂A`/`∂W` (the `SelectLinL` idiom) so they are KA-ready in structure.
2. **Type stability.** `W`/`∂W` are `Vector{Matrix}` (ragged per-`l`); the
   `k`/`l` loops are dynamic. Fine for CPU correctness; for performance, the
   uniform-`K` fast path (§6 "still open") wants a regular layout (padded 3-D
   array + offsets) so the mix is a batched gemm.
3. **Efficient `evaluate_ed`** (forward-over-reverse Jacobian). End-to-end
   *position gradients* (forces) already work through the full graph pipeline
   (graph → EdgeEmbed → pool → CP → energy) via Zygote — validated against
   ForwardDiff to ~1e-15 in `test/formats/cp/test_cp_pipeline.jl`. What's
   deferred is the *efficient* fused Jacobian (the sparse `_jacobian_X` pattern,
   §5), not correctness.
4. **Initializer calibration.** `W`/`λ` use interim fan-in scaling; the proper
   Nth-root law (§6 / `initializers.md`) and its "untrained-output-variance"
   test are not yet implemented.
5. **`whatalloc`/Bumper discipline.** The CPU path allocates with `zeros`; the
   in-place `evaluate!`/`pullback!` + `@no_escape`/`@alloc` discipline (§5) is
   deferred to the KA pass.

Open `[CO]` items unchanged from below: `EquivLinearL` final name/home (it
currently lives in `formats/cp/`, not `utils/`); uniform-vs-ragged `K`;
per-output-`L` `K_L`. The `mb_spec → single-channel` reduction also means the
constructor currently ignores any radial richness *within* an `l` beyond
folding it into `W`'s `n_l` — worth confirming this matches your intent for how
`mb_spec` should specify a TRACE basis.

### PR #130 review decisions (2026-06-17, CO)

- **Rank `K` stays a single global value** (no per-`L`/per-order `K_L`). The
  per-output freedom is already in `λ` (indexed per-`(L, η)`, and `η` carries
  the order); `Āᵏ` are *shared* channels across all `L`/orders, so per-block `K`
  would need a separate `W` per output — breaking the shared-`W`=CP-factor gauge
  and the never-form-`c` cost. Per-output independent CP, if ever wanted, is a
  separate format. The §6 "general per-`L` `K_L` API" idea is **dropped**.
- **`Āspec` moved into the `EquivLinearL` mixer** (it is the mixer's output
  spec); the redundant `CPACEbasis.Āspec` field is removed (still in `meta`).
- **`EquivLinearL` relocation to a shared folder / `utils/` is postponed** to the
  channel-interface step (its API will move from gather-tables to a graded
  input/output leg then) — kept in `formats/cp/` for the prototype.
- **General `et_` initialisers**: `EquivLinearL` (`W`) and `CPACElayer` (`λ`) now
  take an `init` kwarg (`(rng, dims...) -> array`, default fan-in-scaled
  `et_normal`); `cp_equivariant_tensor` forwards `init` to the mixer.
- **Performance (deferred to the perf pass)**: the per-`k` loop in `_cp_evaluate`
  instantiates the sparse `AA` basis each rank. For TRACE the `AA` basis should
  **never be instantiated** — fuse the single-channel `∏ₜ Āᵏ` + carrier
  contraction into one (KA) kernel. JSON handles the Unicode `meta` keys
  (`Āspec`/`𝔸spec`) fine; `meta` is not currently serialised.

---

## 1. Scope & relationship to ACE / MACE / TRACE

The forced architecture (`eqcp.qmd`, summarised in `restructure.md` §1):
*every* manifestly-equivariant finite-rank format is

```
T = Σ_{l,τ}  C^τ_{l1…lN}  ⊗  c^{l,τ}      (fixed CG carrier ⊗ free coeffs)
```

with equivariance living **entirely** in the fixed carrier `C` and all
compression living **entirely** in the `G`-trivial coefficient tensor `c`.
A "format" in ET is therefore a *choice of compression of `c`* + a
*contraction strategy* against pooled `A`; the carrier machinery is shared.

- **ACE** stops at the carrier: it fits unconstrained linear `c` on the
  B-basis `B_nlη = Σ_m C^{lη}_m A_nlm` and never forms `T`
  (`background.qmd`, *Connection to ACE*).
- **TRACE** = ACE's carrier (Stage 1) + a Schur-admissible channel mixing
  `W` on the multiplicity index (Stage 2) + an *ordinary* symmetric CP /
  S-HOPM decomposition of the now-`G`-trivial `c` (Stage 3). `eqcp.qmd`
  proves this 3-stage shape is **canonical and essentially unique** (Schur
  + connectedness): the only freedom is *which* ordinary low-rank format
  sits in Stage 3 (CP here; Tucker → `eqtucker.qmd`), plus the gauge
  choices in §2.
- **MACE** is the *modelling context*, not a tensor format. It wraps the
  same A→(product/coupling)→B construction inside a **message-passing**
  layer: per layer, messages `m_i = Σ W·B` (MACE eq. 11) and the product
  basis `B = Σ_lm C^{LM} ∏_ξ Σ_k̃ w_{kk̃l_ξ} A_{k̃l_ξm_ξ}` (MACE eq. 10).
  MACE's `w_{kk̃l_ξ}` channel mixing inside the product **is exactly the
  Stage-2 `W`** of TRACE, and MACE's "loop tensor contraction" (Alg. 1)
  is the same never-form-`c` evaluation trick used here. MACE adds two
  things that are **OUT of scope** for this format: (a) message passing /
  multiple layers (receptive-field growth decoupled from body order), and
  (b) learnable readouts/MLP heads stacking layers. Those belong to a
  downstream `models/` effort, not to `formats/cp/`.

**What ET implements (this plan):** a single-layer, O(3)-equivariant
**TRACE format** — consume pooled `A` (one node's features), apply the
equivariant carrier + Schur channel mixing + symmetric CP, emit equivariant
features per output `L`. Stackable into MACE-style models later, but the
format itself knows nothing about graphs or layers beyond what the existing
sparse format already does (it consumes `A` the same way).

---

## 2. Math recap (link, don't re-derive)

Full derivation: `eqcp.qmd` (the *resolution* and *identifying with TRACE*
sections). In ET notation, the three stages are:

**Stage 1 — fixed CG carrier.** `C^{lη}_m` (= `C^τ_{l1…lN}` of
`background.qmd`, with `η ↔ τ` the coupling-tree label) spans the trivial
isotypic component `Inv(ρ^{l1}⊗…⊗ρ^{lN})`. Built once, data-independent,
manifestly equivariant. In ET this is the `symmetrisation_matrix` output
(per output `L`), already `S_N`-symmetrised over the `(n,l)` blocks.

**Stage 2 — Schur channel mixing.** For each output channel `k = 1…K`,

```
Ā^l_{k m} = Σ_n W^l_{k n} A_{n l m}          (per-l mixing; identity on m)
```

Schur's lemma forces the equivariant linear map to act *only* on the
multiplicity index `n`, as identity on the irrep components `m`, and
**block-diagonally in `l`** (it may not mix different irrep types). Within
each `l`-block the multiplicity space is the `n`-index, so the general
admissible map is an **independent matrix `W^l ∈ R^{K × n_l}` per `l`** — not
a single `l`-independent `W`. Sharing `W` across `l` is only a parameter-tying
special case you would impose deliberately; the free (and correct default)
form is per-`l`, which is exactly the extra freedom you flagged. This matches
**MACE**: its channel-mixing weight `w_{kk̃l_ξ}` (eq. 10) carries the `l_ξ`
index — i.e. MACE implements it as `W_{knl}` (per-`l`), not `W_{kn}`. (TRACE
eq. (6) `Ā_{i,klm} = Σ_zn W^k_zn A_{i,znlm}` writes a single `W^k`, but its
multiplicity range `zn` is taken per irrep, so it is the same per-`l` object;
species `z` is folded into `n` per `restructure.md` §9: `n` = "everything
invariant".)

**Stage 3 — symmetric CP of the `G`-trivial `c`.** The coefficient tensor
`c_η` (order `ν = N`, `G`-trivial, `S_N`-symmetric) is expanded rank-`K`:

```
c_η = Σ_{k=1}^K λ_{kη}  w_k ⊗ w_k ⊗ … ⊗ w_k     (ν factors)
```

(`eqcp.qmd` Stage 3; TRACE eq. 8–9). The CP factor `w_k` **is** the
Stage-2 mixing — i.e. `W`'s rows *are* the shared CP factors. This is the
gauge identification of §2.G below: TRACE merges Stage 2 and Stage 3 into
one shared `W`, so that the rank index `k` of CP and the channel index `k`
of `Ā` coincide. The free parameters are then `λ_{kη}` (and `W`).

**The never-form-`c` evaluation identity (the whole point).** Substituting
the CP form of `c` into `F = Σ_η c_η · (C^{lη} : A^{⊗ν})` and using that
`Ā^k = W_k · A` is rank-1 across slots, the contraction factorises so `c`
is *never assembled*:

```
F_LM = Σ_η  λ_{·η-block}  ·  Σ_m C^{lη}_{m}  ∏_{t=1}^ν  Ā^{k}_{l_t m_t}
     = Σ_k Σ_η λ_{kη}  B̃^k_{η,LM}
   B̃^k_{η,LM} = Σ_m C^{LM}_{η,m}  ∏_{t=1}^ν Ā^k_{l_t m_t}
```

i.e. **per rank `k`** form the channel-mixed `Ā^k = W_k·A`, build the
*ordinary scalar* product/coupling `B̃^k` against the shared carrier
(exactly the sparse format's `𝔸 → 𝔹` step, but on a single mixed channel),
then take the linear combination over `(k,η)` with weights `λ`. Cost is
`O(K · |carrier|)` instead of `O(n^ν · |carrier|)` — the TRACE/MACE
scaling win (TRACE: `O(K)` vs `O(N^ν S^ν)`; MACE Alg. 1). Indices:
`t = 1…ν` slots, `m = (m_1…m_ν)`, `L,M` the output irrep, `η` the coupling
multiplicity for the `(L, l_1…l_ν)` block.

**Parity / gauge caveats** (`eqcp.qmd`, *free choices*) — record, do not
implement first cut:

- **(G1) coupling tree.** Different `τ` give equivalent (Racah/6j-related)
  carrier bases — a basis choice handled inside `groups/`, invisible to CP.
- **(G2) `W` vs the CP factor `w`.** `W` (Stage 2) and `w_k` (Stage 3) are
  the same linear object; "compress-then-decompose" vs "merged" is a
  parametrisation choice. ET should adopt the **merged** form (`W`'s rows
  = CP factors), which is also what makes the per-`k` evaluation above
  clean. See §5 fields.
- **(G3) `W` vs radial fold.** Pooling-linearity `W·A = pool(W·R · Y)`
  means `W` is equivalently a learned radial basis (`restructure.md` §6.1):
  pure overparameterisation if both ET-`W` and a radial `W` are learnable.
  ET owns the post-pooling primitive; ACEradials owns the fold for
  deployment. Constructors should steer toward one learnable side (§5,§6).
- **(G4) parity twist.** For even `ν`, a `Z_2 = G/G⁰` pseudo-scalar twist
  can pair terms (`eqcp.qmd` point 3); neither TRACE nor this plan exploits
  it. Park.

---

## 3. Mapping onto ET infrastructure

| Stage | What it is | ET home | reuse / new |
|------|-------------|---------|-------------|
| input | pooled `A_{nlm}` | `pooling/sparseprodpool*.jl` (`PooledSparseProduct`) | **reuse as-is** |
| 1 | CG carrier `C^{lη}` per `L` | `groups/symmop.jl` `symmetrisation_matrix`, `groups/O3` `coupling_coeffs` | **reuse**; expose as shared carrier infra (`restructure.md` §6) |
| 2 | Schur mixing `Ā^k = W·A` | equivariant learnable linear layer (ET primitive, `restructure.md` §6.1) | **new** thin layer; `selectlinl.jl` is a partial template (KA matmul + rrule), not a drop-in |
| 3a | per-`k` scalar coupling `B̃^k` | the sparse format's `𝔸→𝔹` contraction, single channel | **reuse pattern** (`SparseSymmProd` + `A2Bmap` mul), **new** wiring for the `k`-loop |
| 3b | CP combine `Σ_k λ_{kη} B̃^k` | small contraction / readout | **new** (a `λ`-weighted reduction; rrule like `_tupmul`) |

Notes on reuse:

- **Carrier is already centralised.** `symmetrisation_matrix` lives in
  `groups/symmop.jl` (not buried in the sparse format) and returns
  `(𝔸2𝔹::SparseMatrixCSC, 𝔸spec)` per `L`; CP calls the **same** function
  with the **same** `mb_spec`. `restructure.md` §6 worried this was buried;
  it is not — only the *spec plumbing* in `sparse_ace_utils.jl`
  (`_make_idx_*`, the `𝔸spec` union/re-index) needs lifting to a shared
  `formats/carrier.jl` (or `specs/`) so both formats share it. **This is
  the one genuine refactor CP needs from Stage 1.**
- **Stage 2 `W` ≠ `SelectLinL`.** `SelectLinL` selects `W[:,:,cat(x)]` by a
  *categorical input* and acts on a flat feature matrix; CP's `W` is a
  *single* (or per-`l`) matrix applied to per-`l` blocks of `A`, with `k`
  as the rank index. Borrow its KA matmul kernel + two-kernel pullback
  (`∂P`, `∂W`) idiom; write a dedicated `EquivLinearL` (name `[CO]`).
- **Stage 3a is the sparse contraction, restricted to one channel.** The
  sparse format already does `𝔹 = (C * 𝔸')'` (`sparse_ace_ka.jl`
  `_ka_evaluate`). CP does the same `C * 𝔸'` but with `𝔸` built from
  `Ā^k` rather than the full `A`, looped over `k`. The `SparseSymmProd`
  evaluate/pullback (incl. `_static_prod_ed`) is directly reusable for
  the `∏_t Ā^k` part.


**Categorical variables (where they enter).** Two places only: (a) in the
construction of `A` — species / one-hot channels folded into the `n` index;
handled **upstream** in pooling, *not* a TRACE/CP concern; and (b) in the
**readout**, where the `λ_{kη}` coefficients may be made category-dependent.
That second case — selecting `λ` by a categorical — is exactly what
`SelectLinL` is for. So `SelectLinL` belongs to **Stage 3b** (the `λ`
readout), not Stage 2: this sharpens the "Stage 2 `W` ≠ `SelectLinL`" note
above. Stage 2's `W` is the equivariant `n`-mixing and never sees categories.

---

## 4. A-storage interaction (cross-ref `restructure.md` §4)

CP is the format that **forces the open A-storage decision**. Stage 2
(`Ā^k = Σ_n W_{kn} A_{nlm}`) is a per-`l` matmul on the `n` index; Stage 3a
contracts on `(l,m)` only. Both want the **per-`l` block** view
`A^l ∈ R^{n_l × (2l+1)}`:

- Stage 2 with per-`l` blocks is one `gemm` `W^l (K×n_l) · A^l (n_l×(2l+1))`
  → `Ā^{l} (K×(2l+1))` per `l` — BLAS/KA-friendly, and the natural place
  to allow ragged `n_l` (we *do* use different `n`-range per `l`,
  `restructure.md` §4). A flat `A` forces gather/scatter through the spec
  for every `k`.
- Stage 3a then reads `Ā^k` as `(l,m)`-indexed, which the existing carrier
  `A2Bmap` already addresses by spec.

**Recommendation.** Adopt the *flat-contiguous-storage + lightweight
per-`l` block view* middle ground from `restructure.md` §4: keep `A` stored
flat (so the sparse format and irregular specs are unaffected), and provide
a `block_view(A, l) -> matrix view` (an offsets table from the `Aspec`)
that Stage 2 uses. This commits CP to the block *access pattern* without
forcing a storage change on the rest of ET — and gives `restructure.md` §4
the prototype it asked for ("prototype both access patterns against CP
before committing"). If the view turns out to cost too much on GPU, fall
back to materialising per-`l` blocks once per forward pass. **Decision (CO):
implement the block-view first**; revisit (block-view vs materialised blocks)
later by benchmarking the Stage-2 matmul on GPU — tracked in §6.

---

## 5. Proposed code layout & interfaces

Mirror the sparse trio (`restructure.md` §3). New dir `src/formats/cp/`:

```
src/formats/cp/
  cp_ace_basis.jl   # CPACEbasis type + spec/constructor; OWNS W (Stage 2
                    #   channel mixing) + carrier + per-k contraction
                    #   (Stage 1+3a); evaluate / evaluate_ed / pullback! /
                    #   whatalloc; Lux initial{parameters,states} for W
  cp_ace_layer.jl   # CPACElayer (Lux): OWNS λ (Stage 3b readout) only;
                    #   wraps the basis; readout to features per L
  cp_ace_ka.jl      # KA kernels: per-k channel mixing, per-k coupling,
                    #   λ-combine; batched (node) versions; pullbacks
  cp_ace_utils.jl   # spec generation; shares carrier plumbing with sparse
                    #   (lifted to formats/carrier.jl or specs/)
```

(Resolved per CO: the **basis owns `W`**, the **layer/readout owns `λ`** —
applied in the comments above and the struct/params prose below.)

Also (shared, from §3): lift the carrier spec plumbing out of
`sparse_ace_utils.jl` into `formats/carrier.jl` (or `specs/`) so both
formats build `(Abasis, 𝔸basis-per-k, A2Bmaps, specs)` the same way.

**What `EquivLinearL` is (the Stage-2 primitive).** A small equivariant
learnable linear layer that does *only* the channel mixing
`Ā^l_{k m} = Σ_n W^l_{kn} A_{n l m}`: it takes pooled `A` as per-`l` blocks
`A^l ∈ R^{n_l × (2l+1)}` and, for each `l`, left-multiplies by the learnable
`W^l ∈ R^{K × n_l}` to give `Ā^l ∈ R^{K × (2l+1)}`. Equivariant by
construction (Schur: mixes `n`, identity on `m`, independent per `l`; §2). It
is the moral sibling of `SelectLinL` — same KA matmul + two-kernel
(`∂A`,`∂W`) pullback idiom — but it *mixes* the multiplicity index `n`,
whereas `SelectLinL` *selects* a weight slice by a **categorical** input on a
flat feature matrix. Different jobs: `EquivLinearL` = Stage-2 equivariant
mixing (owned by the basis); `SelectLinL` = Stage-3b categorical `λ` readout
(§3). It lives in `utils/` beside `selectlinl.jl`; Tucker reuses it. Name
still `[CO]`.

### Main type(s)

The **basis owns `W`** (Stage-2 channel mixing is part of *building* the
equivariant features); the **layer owns `λ`** (the Stage-3b readout). The
struct fields below are fixed config — `W` is the basis's *learnable
parameter*, produced by its Lux `initialparameters` (not a struct field).

```julia
struct CPACEbasis{NL, TA, TAA, TSYM} <: AbstractLuxLayer
   abasis     # PooledSparseProduct        (A_{nlm}; reused)
   aabasis    # SparseSymmProd             (∏_t Ā^k; the scalar product)
   A2Bmaps    # NTuple per L: carrier C^{lη}  (Stage 1; reused)
   LL         # NTuple{NL,Int}  output irreps
   lens       # NTuple{NL,Int}  #carrier rows per L (= #η)
   rank       # K
   ord        # ν   (body / correlation order)
   meta       # Dict (specs, n_l offsets for the block view, basis switch)
end
# basis ps = (W = …,)
#   W : Stage-2 mixing = the shared CP factors (gauge G2: merged form);
#       per-l blocks  W^l ∈ R^{K × n_l}   (Schur: acts on n only, per l)
```

`evaluate` returns features per `L` as a `Tuple` (one entry per output
`L`), matching the sparse `𝔹`-basis I/O contract (`restructure.md` §3,
duck-typed). For an invariant model `LL = (0,)`.

**The layer owns the readout `λ`** (mirrors `SparseACElayer` holding `WLL`):

```julia
struct CPACElayer{TB, NLL} <: AbstractLuxLayer
   basis::TB
   rank::Int
   ord::Int
   # nfeatures etc. as in SparseACElayer
end
# layer ps = (λ = …,)
#   λ : Stage-3 coefficients  λ_{kη}  per output-L block (and per feature);
#       may be category-dependent — the SelectLinL readout role (§3)
```

Constructor signature (kwargs, matching `sparse_equivariant_tensor`):

```julia
cp_equivariant_tensor(; LL, mb_spec, Rnl_spec, Ylm_spec, basis,
                        rank::Int, ord = <inferred from mb_spec>)
```

`mb_spec` defines the carrier exactly as for the sparse format (same
`symmetrisation_matrix` call); `rank` = `K`; `ord` = `ν`. `basis = real`
(committed, `restructure.md` §9) but the carrier-level switch stays in
`groups/`.

### Methods to implement

`evaluate`, `evaluate_ed`, `pullback`/`pullback!`, `whatalloc`, and Lux
`initialparameters` / `initialstates`. Follow the sparse format's in-place
Bumper/WithAlloc discipline (`@no_escape` + `@alloc`, `whatalloc` returning
`(T, dims...)`), and expose the kernels to users *only* through ChainRules
`rrule`s (`restructure.md` §3) — users go through Lux + AD.

### Forward algorithm (pseudocode, in-place, per node; batch = add a node loop)

```
# inputs: A  (pooled features for one node), ps.W (per-l blocks), ps.λ
# output: F :: Tuple over output L
function evaluate(layer, A, ps, st)
  @no_escape begin
    Abl  = block_view(A, layer.basis)            # per-l matrices (offsets in meta)
    F    = ntuple(_ -> zero-accumulator-per-L, NL)
    for k in 1:K
      # Stage 2: channel mix this rank  →  Āk_{l m}
      Āk = @alloc(...)
      for l in Ls:  mul!(Āk^l, ps.W[l][k, :]', Abl^l)      # K-row k of W^l
      # Stage 3a: ordinary symmetric product on the single mixed channel
      𝔸k = @withalloc evaluate!(layer.basis.aabasis, Āk)   # ∏_t Āk (SparseSymmProd)
      # Stage 1 contraction: carrier per L  →  B̃k_{η,LM}
      for (iL, L) in enumerate(layer.basis.LL)
        B̃kL = layer.basis.A2Bmaps[iL] * 𝔸k                # = C^{lη} : 𝔸k
        # Stage 3b: accumulate with λ
        F[iL] .+= λ_combine(ps.λ[iL], k, B̃kL)            # Σ_η λ_{kη} B̃k_{η,LM}
      end
    end
  end
  return F, st
end
```

Real work is the `K`-loop; everything inside it is a single-channel instance
of the sparse pipeline. The `k`-loop is the parallel axis on GPU.

### evaluate_ed / pullback sketch

- `pullback!(∂A, ∂F, layer, A, ps)` runs the forward graph in reverse,
  reusing the sparse pieces: for each `k`, `∂B̃k = λ`-adjoint of `∂F`
  (transpose of `λ_combine`, cf. `_tupmul` rrule); `∂𝔸k = A2Bmap' * ∂B̃k`
  (as in `sparse_ace_ka.jl` `_ka_pullback`); `∂Āk = pullback(aabasis,
  ∂𝔸k, Āk)` (reuse `SparseSymmProd.pullback!` with `_static_prod_ed`);
  then Stage-2 adjoint `∂A^l += Σ_k W^l[k,:] ⊗ ∂Āk^l` and `∂W^l[k,:] +=
  ∂Āk^l · (A^l)'` (the `SelectLinL` two-kernel pullback idiom: separate
  `∂A` and `∂W` kernels to avoid GPU races). `∂λ` from the `λ_combine`
  rrule.
- `evaluate_ed` (w.r.t. input features, for position gradients) follows the
  sparse `_jacobian_X` pattern (push `∂A` through `aabasis` via
  `_static_prod_ed`, then the carrier mul) — or simply expose `rrule` and
  let AD compose, deciding per the `restructure.md` §3 differentiation-API
  question.

### GPU / KA story

KA from day one (`restructure.md` §8). Three kernels, all parallel over
`(k, node, …)`: (1) channel-mix `Ā^k = W·A` (per-`l` batched gemm —
KA matmul like `_ka_apply_selectlinl!`); (2) per-`k` symmetric product +
carrier mul (the sparse `_ka_evaluate` restricted to one channel, looped /
batched over `k`); (3) `λ`-combine reduction. Pullbacks: split `∂A`/`∂W`
into separate kernels (unique outputs) exactly as `selectlinl.jl` does.
The `k`-loop should be a kernel dimension, not a host loop, so a single
launch covers all ranks. Use `KernelAbstractions.get_backend(A)` for
dispatch; CPU is the same kernels (KA-on-CPU), per the sparse format.

### CLAUDE.md compliance

No hard-coded float types (thread `T` from inputs / a `T=Float64` kwarg per
`initializers.md` item 3; pull `2l+1` from `groups/`, never inline —
`restructure.md` §9); type-stable (use `@generated` over `ν`/`ORD` like
`SparseSymmProd` and `ntuple` over `L`, as the sparse format does to absorb
the heterogeneous-spec instability); KA-only GPU (no CUDA in shared paths).

### Learnable params & gauge steering

`W` (on the basis) and `λ` (on the layer) are learnable; the carrier `C` and
specs are fixed state (`initialstates`). Gauge redundancy (§2 G2/G3): the
**merged** `W`-as-CP-factor form removes the Stage2-vs-Stage3 double-count
(G2). For G3 (W vs radial fold), the constructor/initializer should steer
toward **one** learnable side — default: fixed/orthonormal radials +
learnable `W`+`λ` for training. The deploy-side fold/compression (ACEradials'
splined radials) is **deferred (CO)** — not needed now, address later (ET
does *not* enforce this, only documents/defaults it — `restructure.md` §6.1).

---

## 6. Initializers, testing, sequencing, open questions

### Initializers (cross-ref `initializers.md`)

CP is the **first multiplicative format** and the concrete consumer the
initializer notes were written for. Implement the **Nth-root law**
(`initializers.md` §1): output `≈ Σ_k ∏_n f_{kn}` ⇒ `Var ~ K·σ^{2ν}` ⇒
`σ ~ (target/√K)^{1/ν}`, applied to the `W`/`λ` leaf draws. `W` must be
**irrep-aware** (`initializers.md` §2): per-`l` blocks acting on `n` only,
dims from `groups/`. Reuse the `et_normal`/`et_zeros` leaf samplers
(`utils/initializers.jl`); eltype `Float64` default, `Float32` only on
explicit request (`initializers.md` §3). Land the format-aware policy
*with* CP (as `initializers.md` recommends), unit-tested via "untrained CP
output std ≈ target across `ν`, `K`".

### Testing plan (style of `tests.md`)

Land under `test/formats/cp/` (mirror `src/`, `tests.md` house rule). Use
the shared `LTM` / `DIFF` helpers (`tests.md`) and `detect_gpu_backend()`
for the GPU lane (CPU no-op on CI).

1. **Correctness vs reference.** For small `ν, K`: build CP, build the
   *same carrier* via `sparse_equivariant_tensor` with the **dense `c`**
   reconstructed from `(λ, W)` (`c_η = Σ_k λ_{kη} ∏_t W_{k·}`), and check
   `CP.evaluate ≈ sparse.evaluate` on random `A`. This reuses the
   "dense = sparse constructor" path (`restructure.md` §6) as the oracle —
   so the dense-via-sparse step (§8.3) should land first. `[CO]`: provide a
   standalone dense reference path inside `formats/cp/` too, or rely on
   sparse? (Recommend: rely on sparse — no duplicate carrier code.)
2. **Equivariance.** `F(g·A) = D^L(g) F(A)` for random `g ∈ O(3)` per
   output `L` (the existing O3 quadrature/rotation test utilities; compact-
   only, fine here).
3. **Finite-difference gradients.** `DIFF.grad_fd ≈ grad_zy` w.r.t. **input
   features** and w.r.t. **parameters** `(W, λ)` — the latter exercises the
   Stage-2/Stage-3 rrules (cf. `tests.md` gap-1 for the sparse layer).
4. **Allocation / Lux / GPU parity.** `whatalloc` correctness (no escapes);
   `initialparameters`/`initialstates` shapes & eltype; CPU≈GPU forward and
   parameter-gradient via `dev()` (F32 always, F64 when `gpu_supports_f64`),
   matching the `test_model.jl` pattern. Position-gradient GPU parity as a
   real `@test`, `@test_broken` on Metal only (`tests.md` Known bugs).
5. **Initializer law** (above): untrained-output variance vs target.

### Sequencing (slot into `restructure.md` §8)

§8 already places CP at step 4, after (2) the format-I/O-contract + A-storage
settling and (3) the dense-via-sparse constructor. Refinement:

1. (§8.2) settle A-storage with the **block view prototyped against CP**
   (§4) — CP is the forcing function, so do a throwaway CP Stage-2 spike
   here.
2. (§8.3) dense-via-sparse — also the CP **test oracle** (testing #1).
3. lift carrier spec plumbing to `formats/carrier.jl` (shared infra, §3).
4. (§8.4) CP format proper: basis → layer → KA, with initializers.
5. Tucker reuses `EquivLinearL` + carrier infra (gated on `eqtucker.qmd`).

### Decisions (CO) & remaining open questions

**Resolved (CO):**

- **A-storage** (§4): **try flat + block-view first**; revisit (block-view vs
  materialised per-`l` blocks) later by benchmarking the Stage-2 gemm on GPU.
  Still *the* key data-structure call (`restructure.md` §4, §10) — just not
  blocking the first cut.
- **W placement / gauge (G2,G3)**: adopt the **merged** `W`-as-CP-factor form;
  default to learnable `W`+`λ` with fixed radials for training. The deploy-side
  fold/compression is **deferred** — not needed now (§5, `restructure.md` §6.1).
- **Ownership**: the **basis owns `W`**, the **layer owns `λ`** (§5).
- **Dense reference path**: **reuse the sparse dense constructor as the test
  oracle** — no duplicate carrier code (§6 testing #1).
- **Parity twist (G4)**: **park** — TRACE doesn't use it (`eqcp.qmd` point 3).
- **Differentiation API**: implement `pullback!` first, then put a ChainRules
  `rrule` on top of it (`restructure.md` §3).

**Still open:**

- **Rank-selection / spec API**: keep the **general** form (per-output-`L` /
  per-`(l-block)` `K_L`) in the constructor, defaulting to a single global `K`.
  Sub-question (CO): does forcing a uniform `K` buy enough performance to
  prefer it? Likely a *mild* win only — a uniform `K` keeps `W`/`λ` regular (no
  ragged shapes), so the `k`-axis is one uniform GPU launch dimension and
  storage stays contiguous; per-block `K_L` needs padding or an offsets table
  and a ragged launch. Plan: expose the general API but make the uniform-`K`
  path the fast default; specialise the ragged path only if benchmarks show it
  hurts.
- **`EquivLinearL` naming + home** (the Stage-2 primitive — defined in §5):
  the name and whether it lives beside `selectlinl.jl` is still `[CO]`
  (`restructure.md` §10).
