# A general interface for equivariant tensors and the maps between them

Status: rev 2 (2026-06-16). Design notes — **prose only, no Julia types yet**.
Rev 2 records decisions: separate single-group legs (no product sectors), raw
inputs = product of legs, `coupling` = the fixed carrier, and the sequencing
(§9: merge TRACE → interface restructure across both formats).
Goal: agree on the *contract* for equivariant tensors in ET before committing to
any API. Trigger: how to access `A_nlm` (the per-`l` block view the CP/TRACE work
needed); broader aim: let equivariant maps compose into tensor networks with
machine-checkable consistency.

This proposes; it does not implement. CO's original framing is in §1; the
refinements and the genuinely open questions are flagged `[open]`. Cross-refs:
`trace.md` (CP/TRACE), `restructure.md`.

---

## 1. Starting point (CO's framing)

> A general tensor `A` has `P` channels `i₁,…,i_P`. Each channel is labelled
> *trivial* or *equivariant*. E.g. for `A_nlm`: `i₁ = n` (trivial),
> `i₂ = (l,m)` (equivariant). How the tensor stores its components is irrelevant
> to the interface. ACE and TRACE are not really tensors — they are **maps
> tensor → tensor** — so they need *two* specifications: input channels
> `(j₁,…,j_{Pin})` and output channels `(i₁,…,i_{Pout})`. This gives (i) easy
> consistency checks and (ii) composable tensor networks.

We agree with the spine of this: **ACE/TRACE are maps, storage is opaque, and a
map is specified by its input and output index structure**. The rest of this doc
sharpens *what an index actually carries*, *what extra a map needs beyond
in/out*, and *how to tell a map apart from the data flowing between maps*.

---

## 2. Legs and the spaces they carry

**Terminology.** Drop `P`. A tensor has labelled **legs** (synonyms in the wild:
indices, modes, bonds). We say "leg".

**Each leg carries a *space*, and a space is graded by *sectors*.** A sector `s`
is an **irreducible representation (irrep) of the ambient symmetry group `G`**. A
leg's space is a direct sum

```
   V  =  ⊕_s  ( mult_s  ×  irrep_s )
```

i.e. for each sector `s` a *multiplicity* `mult_s` (how many copies) times the
irrep's own dimension `dim_s`. This is exactly e3nn's `Irreps` object
(`"32x0e + 16x1o"` = 32 copies of the scalar irrep ⊕ 16 copies of the
odd-vector irrep) and TensorKit's `GradedSpace` (a `Sector`-graded space).

**"Trivial vs equivariant" is a *derived projection*, not the primary label**
[CO item 1, agreed]. Every leg carries a representation; a "trivial channel" is
just the special case where the sector is the trivial irrep (so it transforms by
the identity). Keeping the full sector label, not the binary, is what lets us
*build* the equivariant carrier. But the binary is still useful — it is precisely
what **Schur's lemma** keys on: an equivariant linear map may mix the
multiplicity space of a sector freely, but must act as the **identity on the
irrep components** and **block-diagonally across sectors**. (This is exactly the
TRACE Stage-2 constraint: `W` mixes `n` per `l`, identity on `m` — `trace.md §2`.)

### 2.1 `A_nlm` — and why **inputs** should not eagerly merge `n` into `(l,m)`

There are two genuinely different situations and the interface must keep them
apart [refines an earlier over-merge; CO pushback 2026-06-16].

**(1) `n` has cross-sector meaning — a *shared* radial index.** E.g. CO's
multi-`O(3)` embedding `R_{n l1 l2}(|r|,|m|,z) · Y_{l1 m1}(r̂) · Y_{l2 m2}(m̂)`:
`n` is the *same* space for every sector, merely modulated by it. Then `n` is its
own **trivial leg** and the input is a clean **product of legs**

```
   A :  n(trivial) ⊗ (l1,m1)[O(3)_r] ⊗ (l2,m2)[O(3)_m] ⊗ z(trivial)
```

— one equivariant leg per independent group factor, plus trivial legs for the
radial multiplicity and the species. The `(l1,l2)`-dependence of `R` is
**content, not structure**: it populates the legs without changing the layout.
*Here there is no value in coupling `n` to `(l1,l2)`* — the product is more
transparent (you can read off the `O(3)×O(3)` action) and composes (a consumer
checks `irreps_in` factor-by-factor). Merging would hide exactly that structure.

**(2) `n` is entangled with the sector — the traditional ACE `R_{nl}`.** The
radial basis at degree `l` has no shared meaning across `l`, and pruning keeps a
different count per `l`. Then there *is* no shared `n` leg, and the honest model
is a single **graded leg** `⊕_l (n_l × irrep_l)` with a **per-sector
multiplicity** `n_l`. The per-`l` "block view" used in the CP code
(`cp_ace_basis.jl`: `mixer.nl_count`, `mix_Acols`) is then "iterate the leg by
sector". This is the only place the merge earns its keep on the *input* side.

So merging `n` into a flat `(n,l,m)` is appropriate for **(a) coupled features**
(no product structure survives coupling — this is exactly how the sparse linear
ACE layer is already built) and **(b) internal storage** of sector-entangled
radials — but it should **not** be the default for a **raw particle embedding**,
which should preserve the product of legs. The interface must therefore support
*both* `n`-as-trivial-leg (shared) and `n`-as-per-sector-multiplicity (entangled).

**Stacked layers (MACE) — the same model, no special case.** When equivariant
layers are stacked (MACE-style), the *output* of one layer — a coupled feature
`(l,m)` with multiplicity `n` (the channel index) — is the *input* of the next.
So the merged graded leg is a perfectly legitimate **input value**, not an
"output-only" shape. The honest distinction is therefore *raw particle embedding*
(product of legs, case 1) vs *coupled feature* (merged graded leg) — and **both
are valid values**; which one a layer accepts is just its declared `irreps_in`,
settled by the §3 consistency check. Nothing special is needed, and internal
memory layout stays free to optimise. (Per CO, re-implementing MACE is **not** a
goal — the point is only that the interface already accommodates stacking; the
priority is trying out new *formats*.)

⚠️ ET is *more general* than e3nn either way: e3nn's `mul × irrep` is
rectangular, whereas ET's `mb_spec` selects an **arbitrary sparse subset** — see
§4 (sparsity).

---

## 3. Values vs operators: where the domain/codomain split lives

CO item 3: *"a map is a tensor with a domain/codomain split — but they must be
clearly split, and the output of a map (input to the next layer) is still a
'simple' tensor without such a split. How do you distinguish?"*

The clean answer (TensorKit's view): **the distinction is the presence of a
domain.**

- A **value** (a "state") is a *domain-less* tensor: every leg is on the output
  side (the codomain). It is just an `Irreps`-labelled array — the data that
  flows between layers. `A`, and the `𝔹`/feature output, are values.
- An **operator / map** (a layer; the carrier `C`) is a tensor whose legs are
  **split** into a **domain** (`irreps_in`, the legs it consumes) and a
  **codomain** (`irreps_out`, the legs it produces).
- **Applying** an operator contracts its domain against a value's legs and
  returns a new **value** (domain-less again):

```
   operator :  irreps_in  ──▶  irreps_out          (has a domain)
   value    :  ()          ──▶  irreps              (domain empty)

   apply(operator, value)  :  value'                (domain-less)
        └── requires  value.irreps  ≟  operator.irreps_in     (the consistency check)
```

So a "simple tensor between layers" is the special case `domain = ()`; a layer is
the case `domain ≠ ()`. A pipeline is `apply(L_k, … apply(L_2, apply(L_1, A)))`,
and each `apply` checks `irreps_out(L_i) == irreps_in(L_{i+1})`. That is the
composability + consistency CO wants, and it falls straight out of "a map is a
tensor with a domain". (A map can equivalently be *re-read* as a value by
"bending" all its legs to the codomain — same data, the tensor–hom adjunction —
which is why we need only **one** underlying notion, a leg-labelled tensor, plus
a domain/codomain partition.)

---

## 4. A map needs a *coupling* spec, not just in/out

CO item 4 — the key piece missing from the original framing, and tied to the
product-group concern below. Input and output `Irreps` **do not determine** an
ACE/TRACE map. ACE is a **degree-`ν` symmetric multilinear** map: the input value
appears `ν` times (body order), symmetrised. The full signature is

```
   ( irreps_in ,  coupling ,  irreps_out )
```

where **`coupling`** says *which input multi-indices fuse to which output
irreps*. This is where the Clebsch–Gordan / fusion coefficients live, and where
**sparsity** lives. In ET, `coupling` **is `mb_spec`** — the list of `(n,l)`
blocks to keep. (e3nn calls this `instructions`; NVIDIA cuEquivariance calls them
"segmented tensor-product descriptors".) Two maps with identical `irreps_in` and
`irreps_out` but different `coupling` are different maps.

**`coupling` is the *fixed structural* part; learnable parameters are separate.**
In the forced architecture `T = Σ C⊗c` (`trace.md §1`), `coupling` is the **fixed
carrier `C`** (the CG/fusion coefficients + *which* products to form = `mb_spec`).
The **learnable** part — the coefficient tensor `c` (sparse: the readout weights;
CP/TRACE: the mixing `W` and CP weights `λ`) — is the map's **parameters**, not
its signature. So `(irreps_in, coupling, irreps_out)` is the *type* of the map;
the parameters fill it in. One `coupling` concept must cover both formats:
sparse = `mb_spec` + `symmetrisation_matrix`; CP = `mb_spec` + rank `K` + the
single-channel reduction (`trace.md`). Confirming this uniformity across the two
formats is exactly the co-design payoff of the sequencing in §9.

**"A map is a tensor" vs ACE being nonlinear.** ACE/TRACE are degree-`ν`, so not a
single *linear* `C` applied to `A`. The reconciliation: the **equivariant content
is a linear carrier on the symmetric `ν`-th power** `𝔸` (the pooled products),
and forming `𝔸` from `A` is a **fixed structural functor** (the symmetric power;
in ET, `SparseSymmProd`), not a learnable map. So the domain/codomain picture
(§3) applies to the linear carrier (domain = `𝔸`-space, codomain = output
`L`-space); the polynomial step lives in `coupling`-structure, not on a leg.

**Sparsity `[open]`.** Three different sparsities, to be placed deliberately:
1. *ragged multiplicities* (`mult_s` varies by sector, e.g. fewer `n` at high
   `l`) — lives **in the leg's graded space**; fine.
2. *sparse coupling* (only some `(n₁l₁,…,n_νl_ν)` products kept) — lives **in the
   coupling spec**; fine, this is `mb_spec`.
3. *genuinely arbitrary within-leg subsets* (a specific `(n,l)` present but a
   "neighbouring" one absent in a way `mult_s` can't express) — **decide**
   whether the interface must represent this in the leg, or whether it is always
   expressible as (1)+(2). My current read: (1)+(2) cover ET's cases, but this
   needs confirming against the real `mb_spec` patterns.

---

## 5. Product groups: independent vs joint transformation `[open — the crux]`

CO item 2: *"what about `(n, l1, l2)` where `l1, l2` transform **independently**,
not jointly — e.g. an `O(3)×O(3)` product group? How does that fit?"*

It fits cleanly **iff the sector label is "an irrep of `G`", with `G` allowed to
be a product group** — not "a single `l`". Then:

- **Independent factors → one leg per factor (the transparent default).** CO's
  `… Y_{l1 m1}(r̂) Y_{l2 m2}(m̂)` is already the *factored* form: two separate
  equivariant legs, one per `O(3)`, each graded over its own `l`. `O(3)×O(3)`
  acts factor-wise; the legs never mix. This is the most readable input layout
  and is exactly the embedding the user writes (§2.1 case 1).
- **Equivalently, fuse to product-sectors when wanted.** The same space can be
  carried by *one* leg graded over **product-group irreps** `ρ^{l1} ⊠ ρ^{l2}`
  (sector = the pair `(l1,l2)`, dim `(2l1+1)(2l2+1)`). TensorKit's
  `ProductSector`/`⊠` is exactly this; e3nn (single `O(3)`) cannot express it.
  Fusion is a *choice* (e.g. for a map that couples the two factors), never
  forced on the input.
- **Joint coupling** (`l1, l2 → L` via CG within *one* `O(3)`) is **not a leg
  property at all** — it is an operation a *map* performs, and so it lives in the
  **coupling spec** (§4), not in the sector label.

So the doc's central structural claim:

```
   how a leg TRANSFORMS    →  its sectors (irreps of G; G may be a product group)
   how a map FUSES legs    →  its coupling spec  (CG / fusion; + sparsity)
```

Separating these two is what makes both `O(3)`-ACE and an `O(3)×O(3)` (or
`O(3)×Sₙ`, `O(3)×U(1)`, …) construction expressible in one interface. The
`(n, l1, l2)`-independent case is **separate single-group legs** (one per
factor); an `(n, l1, l2)`-jointly-coupled case is "a map whose coupling fuses
`l1,l2`".

**Decision (CO, 2026-06-16): use separate single-group legs + coupling; park
product-group sectors.** Independent factors stay as separate `O(3)` legs and the
sector/`Irreps` object stays single-group — simpler, and it covers the
spin×position example cleanly. Product-group sectors (`⊠`, TensorKit
`ProductSector`) are recorded only as the escape hatch should a future format
genuinely need a single *fused* `(l1,l2)` leg.

---

## 6. Grounding in the current ET code

The interface is a *renaming + unification* of structures that already exist as
bare `NamedTuple`s / `SVector`s — there is currently **no** first-class irrep /
space / sector type (the only structured container is `SYYVector`).

| Interface notion | Current ET realisation | Where |
|---|---|---|
| graded leg of `A` (`⊕_l n_l × irrep_l`) | `Aspec :: Vector{@NT{n,l,m}}` (flat (n,l,m) list) | `sparse_ace_utils.jl:43` |
| sector iteration / block view | per-`l` `mix_Acols`, `nl_count` | `cp_ace_basis.jl` (mixer) |
| coupling spec | `mb_spec :: Vector{Vector{@NT{n,l}}}` | `sparse_ace_utils.jl`, `symmop.jl:11` |
| the carrier as an in→out **morphism** | `A2Bmap` sparse matrix: **cols = input `𝔸` space, rows = output `𝔹/L` space** | `symmop.jl:80` |
| output value with a graded `L`-leg | tuple over `L`; `LL`, `lens`; `L>0` carried as `SVector{2L+1}` | `sparse_ace_basis.jl:8-12, 22-23`; `symmop.jl:48` |
| a sector-block container (one `L`) | `SYYVector{L,N,T}` (`N=(L+1)²`), indexable by `(l,m)` or `l`-block | `groups/O3/yyvector.jl:5-7,68` |

Notably the carrier `A2Bmap` is **already exactly** the "map = tensor with input
space (cols) and output space (rows)" object — the interface would just give its
domain/codomain explicit, checkable labels instead of recovering `L` from a
matrix dimension (`(length(A2Bmap[i][1]) - 1) ÷ 2`, `sparse_ace_basis.jl:22-23`).

---

## 7. Prior art (what to borrow, what to avoid)

- **e3nn / e3nn-jax** — `Irreps` (`"32x0e+16x1o"`); layers spec'd by
  `irreps_in1, irreps_in2, irreps_out` + `instructions`. *Borrow:* the
  `Irreps` = list of `(mul, irrep)` leg label, and `irreps_in/out` naming. *Gap:*
  single `O(3)` only (no product sectors); rectangular `mul × irrep` (no sparse
  subsets). Docs: <https://docs.e3nn.org/en/stable/api/o3/o3_irreps.html>,
  <https://docs.e3nn.org/en/stable/api/o3/o3_tp.html>.
- **TensorKit.jl** — `ElementarySpace`/`GradedSpace` parametrised by `Sector`;
  `TensorMap` is literally `codomain ← domain`; `ProductSector`/`⊠` for product
  groups; leg-bending unifies states and operators. *Borrow:* the whole
  value/operator (domain/codomain) model and `Sector` (incl. product sectors).
  Docs: <https://jutho.github.io/TensorKit.jl/stable/man/spaces/>,
  <https://jutho.github.io/TensorKit.jl/stable/man/tensors/>.
- **ITensors.jl / TeNPy** — `Index`/`LegCharge` carrying QN/charge **sectors**
  (abelian); map is *implicit* via index matching. *Borrow:* the leg-carries-
  sectors idea; *avoid* the implicit-matching map model (we want an explicit
  domain/codomain + coupling). <https://itensor.github.io/ITensors.jl/stable/>,
  <https://tenpy.readthedocs.io/en/latest/intro/npc.html>.
- **NVIDIA cuEquivariance** — "segmented" tensor products = our `coupling`/
  instructions, GPU-oriented. <https://github.com/NVIDIA/cuEquivariance>.

**Naming to adopt** [CO item 5, agreed]: `Irreps` / `irreps_in` / `irreps_out`
(e3nn, familiar to the ML audience) over the **domain / codomain / sector**
mental model (TensorKit, the principled core). Avoid "P-channel".

---

## 8. Decisions & remaining residuals

1. **Input leg layout & product groups** (§2.1, §5) — **decided (CO,
   2026-06-16)**. Default raw inputs to a **product of legs** (one equivariant
   leg per independent group factor + trivial legs for shared `n`, species `z`);
   reserve the **merged graded leg** (`⊕_l n_l × irrep`) for sector-entangled
   radials and coupled features (incl. stacked-layer inputs). Independent factors
   are **separate single-group legs** + coupling; **no native product-group
   (`⊠`) sectors** for now (parked). The sector/`Irreps` object stays
   single-group. *Residual (implementation-time, no abstraction change):* the
   mechanics of supporting both `n`-as-trivial-leg (shared) and
   `n`-as-per-sector-multiplicity (entangled).
2. **Sparsity placement** (§4) — confirm ET's real `mb_spec` patterns are covered
   by ragged multiplicities (in the leg) + sparse coupling, or whether arbitrary
   within-leg subsets must be representable on the leg itself.
3. **Naming finalisation** — leg vs index vs mode; keep e3nn surface + TensorKit
   core?
4. **When to introduce Julia types** — **decided: see §9 (sequencing).** Likely
   shape: a `Sector`/`Irreps` leg label, a value vs. operator type carrying
   `irreps` / `(irreps_in, coupling, irreps_out)`, and a checked `apply`/
   `compose`. A migration re-expresses `Aspec` as a graded leg, `mb_spec` as
   coupling, and `A2Bmap` as an explicit morphism.

---

## 9. Sequencing (decided, CO 2026-06-16)

**Merge TRACE first, then restructure onto this interface using the sparse *and*
CP formats as joint design targets** — not the reverse (interface-first, then
rebase TRACE). Rationale:

- A generality-claiming abstraction is only validated by **≥2 real consumers**.
  Designing it against the sparse format alone risks baking in sparse-specific
  assumptions — the very thing the interface should transcend. CP in-tree is the
  second consumer that proves or breaks it (and resolves the §8 residuals + the
  "one `coupling` across formats" question in §4).
- It **banks completed, validated work** (PR #130) and avoids re-validating TRACE
  on a moving foundation. CP already exists on the current specs, so
  interface-first buys nothing CP-specific.
- The interface is **mostly additive**: `A2Bmap` already *is* the morphism,
  `Aspec` the leg, `mb_spec` the coupling (§6). So "later" is cheap.

**Incremental, not big-bang.** Start with a thin vertical slice — the **A-access
/ graded-leg block view** (the original trigger, needed by both formats) —
co-designed against sparse + CP; prove it; then expand to the value/operator +
coupling layer. Concrete order:

1. Merge TRACE (#130) after CO review.
2. Interface slice: the graded-leg / sector block-view for `A`, expressed for
   both sparse and CP (replaces the ad-hoc `mix_Acols`/`nl_count` in
   `cp_ace_basis.jl` and the flat `Aspec` access).
3. Expand: `Irreps`/value/operator types + checked `apply`; re-express
   `mb_spec`→coupling and `A2Bmap`→explicit morphism across both formats.
