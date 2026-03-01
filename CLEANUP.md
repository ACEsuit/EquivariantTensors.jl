# EquivariantTensors.jl — Cleanup Plan

Generated: 2026-02-28

---

## Checklist

- [ ] 1. Delete `simpletrans.jl`
- [ ] 2. Delete commented `_pfwd` block in `frules.jl`
- [ ] 3. Move live frule into `sparse_ace_basis.jl` and delete `frules.jl`
- [ ] 4. Merge `NTtransformST` / `TransformST`, remove aliases
- [ ] 5. Replace NB=1–4 copy-paste pullback! with `@generated`
- [ ] 6. Fix GPU atomic writes in `SelectLinL`
- [ ] 7. Replace `Meta.parse` string generation with `Expr` in `diffnt.jl`

---

## Details

### 1. Delete `simpletrans.jl` — DEAD CODE

**File:** `src/transforms/simpletrans.jl`

The file opens with:
```
# NOTE: this is currently not part of the package
# TODO: decide whether to keep it or remove it.
```
It is not included in the main module. Contents:
- `Get{SYM}` struct — has a latent bug on line 21: broadcasts over `x` instead
  of `X` (wrong variable). Not exported, not reachable.
- `IDtrans` struct — fully commented out with inline comment "we probably don't
  need this anymore".

**Action:** Delete the file entirely.

---

### 2. Delete commented `_pfwd` block in `frules.jl` — DEAD CODE

**File:** `src/frules.jl`, lines 96–170

A large `#= ... =#` block containing `_pfwd` functions that operate on
`Polynomials4ML.PooledSparseProduct` and `Polynomials4ML.SparseSymmProdDAG`.
These are dead code from an earlier design where those types lived in
Polynomials4ML. The file header itself says:
```
# Moved some frules and pushforwards out of the way of the main files
# because I believe them to be unnecessary.
```

**Action:** Delete lines 96–170 (the `#= ... =# ` block).

---

### 3. Move live frule into `sparse_ace_basis.jl`, delete `frules.jl`

**Files:** `src/frules.jl` (lines 1–93), `src/ace/sparse_ace_basis.jl`

The live `frule` for `SparseACEbasis` and its two helpers
(`_pushforward_abasis!`, `_pushforward_aabasis!`) live in a top-level
`frules.jl` instead of alongside the type they extend. This means reading
`sparse_ace_basis.jl` gives no indication a frule exists.

**Action:** Move lines 1–93 of `frules.jl` into `sparse_ace_basis.jl`
(after the type definition). Delete `frules.jl`. Remove its `include` from
the main module file.

---

### 4. Merge `NTtransformST` / `TransformST`, remove aliases

**Files:** `src/transforms/decpart.jl`, `src/transforms/sttrans.jl`

`NTtransformST` has three names:
- `NTtransformST` — the struct
- `dp_transform` — the constructor
- `nt_transform = dp_transform` — a bare global variable alias (won't
  forward new methods correctly)

`TransformST` / `st_transform` is nearly identical:
- Same fields: `f::FT`, `refstate::ST`
- Same `AbstractLuxLayer` inheritance
- Same `initialparameters` / `initialstates`
- Only difference: `NTtransformST` restricts input to `NTorDP` and has
  `evaluate_ed` / `_pb_ed` via `DiffNT.grad_fd`; `TransformST.evaluate_ed`
  has a commented-out broadcast path silently replaced with `map`.

**Action:** Determine whether `TransformST` has any callers outside the
package. If not, delete `sttrans.jl` and remove its include. Rename
`NTtransformST` to something clearer (e.g. `DPTransform` or just keep
`NTtransformST` but drop the `nt_transform` alias). Consolidate constructor
under one name.

---

### 5. Replace NB=1–4 copy-paste `pullback!` with `@generated`

**File:** `src/ace/sparseprodpool.jl`

Four hand-written specializations of `pullback!` for `NB = 1, 2, 3, 4`
(~130 lines of near-identical code). A generic version already exists and is
correct; a comment says it "seems to be slow" without a benchmark reference.

Problems:
- A logic fix must be applied 4 times
- Performance claim is unverified
- Existing comment even says "TODO: revert to code generation"

**Action:** Benchmark the generic vs. specialised versions. If specialisation
is genuinely needed, replace with a single `@generated` function. If the
performance difference is negligible, delete the specialisations and keep
only the generic.

---

### 6. Fix GPU atomic writes in `SelectLinL` — CORRECTNESS BUG

**File:** `src/utils/selectlinl.jl`

The pullback kernel performs non-atomic writes to `∂W`. Multiple threads can
write to the same location — this is a data race on GPU. A TODO comment in
the file acknowledges this:
```
# TODO: GPU atomic write issues in pullback
```

**Action:** Add `@atomic` to the relevant write(s), or restructure to
avoid the race (e.g. one thread per output element). Add a test that runs
the pullback on GPU and checks correctness.

---

### 7. Replace `Meta.parse` string generation with `Expr` in `diffnt.jl`

**File:** `src/transforms/diffnt.jl`

Several `@generated` function bodies are constructed via `Meta.parse()` with
string interpolation. This works but is fragile — syntax errors appear at
runtime, not compile time, and the code is hard to read.

**Action:** Rewrite the code-generation expressions using `Expr(...)` and
`QuoteNode` directly. Behaviour should be identical; the result is more
robust and inspectable.

