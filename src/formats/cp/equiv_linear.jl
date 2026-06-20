#
# EquivLinearL — the Stage-2 equivariant channel-mixing primitive (trace.md §5).
#
# Maps pooled features A_{nlm} to K mixed channels, mixing *only* the
# multiplicity index n, block-diagonally in l, identity on m (Schur's lemma):
#
#   Āᵏ_{lm} = Σ_n Wˡ_{kn} A_{nlm}
#
# Equivariant by construction. It is the moral sibling of `SelectLinL` (same
# matmul-style mixing + a two-part `∂A`/`∂W` pullback), but it *mixes* the
# n-index rather than *selecting* a weight slice by a categorical. The learnable
# parameter `W` is a per-l block `Wˡ ∈ R^{K × n_l}`; the layer stores only the
# (data-independent) gather tables that map the flat A to the mixed channels,
# plus the output (mixed-channel) spec `Āspec`.
#
# `CPACEbasis` owns `W` and drives this primitive per rank; Tucker will reuse it.
# Its home (`formats/cp/` vs a shared folder / `utils/`) and its API (gather
# tables vs a graded input/output leg) are revisited when the channel interface
# lands — see agents/eqtensor_interface.md and trace.md §0.
#

import ChainRulesCore: NoTangent, rrule, unthunk
import LuxCore: AbstractLuxLayer, initialparameters, initialstates
using LinearAlgebra: dot
using Random: AbstractRNG


"""
   struct EquivLinearL

Equivariant linear "channel-mixing" layer. It maps a matrix of pooled input
features `A` to `K` mixed output channels `Ā` by taking, separately within each
angular block `l`, a learnable linear combination of the input radial channels —
the *same* combination for every component `m` of that block. This is the only
form an equivariant linear map can take (Schur's lemma): it may mix freely within
a multiplicity (radial) space, must be block-diagonal across degrees `l`, and
acts as the identity on the `2l+1` components of each `l`.

### Arrays

- input  `A`  : `(nnodes, nA)` matrix — column `p` is one input feature, labelled
  by an `(n, l, m)` triple (radial channel `n`, degree `l`, order `m`).
- params `W`  : a `Vector` of matrices, `W[il]` of size `(K, n_l)` — one block
  per distinct `l` (`il = 1 … length(nl_count)`), with `n_l = nl_count[il]` the
  number of input radial channels at that `l`.
- output `Ā`  : `(nnodes, K, nĀ)` array — `Ā[node, k, q]` is mixed channel `k` of
  output slot `q`, where `q = 1 … length(Āspec)` runs over the distinct output
  `(l, m)` pairs (the radial index `n` has been contracted away).

### Index tables (data-independent; built by `cp_equivariant_tensor`)

- `mix_l[q]`     : the `W`-block index `il` for slot `q` — the position of slot
  `q`'s degree `l` in the sorted list of distinct `l`'s. There is one block per
  distinct `l`, so all slots `q` sharing the same `l` share the same `il`.
- `mix_Acols[q]` : the columns of `A` that feed slot `q` — one per radial channel
  `n`, i.e. all columns sharing slot `q`'s `(l, m)`, ordered by `n`. Its length
  is `n_l = nl_count[il]`.
- `Āspec[q]`     : the `(n=1, l, m)` label of output slot `q`.

The only learnable parameter is `W` (`ps.W`); the layer carries no state. See
`_eql_apply` for the exact computation, and `agents/trace.md` /
`agents/eqtensor_interface.md` for its role as the CP/TRACE Stage-2 mixing.
"""
struct EquivLinearL{FI} <: AbstractLuxLayer
   rank::Int                       # K mixed channels
   nl_count::Vector{Int}           # #radial channels n_l per distinct l
   mix_l::Vector{Int}              # for each output slot q: index into the W blocks
   mix_Acols::Vector{Vector{Int}}  # for each q: input columns of A (one per n)
   Āspec::Vector{@NamedTuple{n::Int, l::Int, m::Int}}   # output (mixed-channel) spec
   init::FI                        # weight initialiser, called as init(rng, K, n_l)
end

# default W initialiser: fan-in (n_l) scaling keeps the mixed channel Ā ≈ O(1)
# when A ≈ O(1). Any `(rng, dims...) -> AbstractArray` works (e.g. `et_zeros`,
# or `(rng, d...) -> et_normal(rng, d...; σ = 0.1)`).
_eql_default_init(rng::AbstractRNG, dims::Integer...) =
      et_normal(rng, dims...; σ = inv(sqrt(dims[end])))

EquivLinearL(rank, nl_count, mix_l, mix_Acols, Āspec; init = _eql_default_init) =
      EquivLinearL(rank, nl_count, mix_l, mix_Acols, Āspec, init)

Base.length(l::EquivLinearL) = length(l.Āspec)

Base.show(io::IO, l::EquivLinearL) =
      print(io, "EquivLinearL(rank = $(l.rank), nblocks = $(length(l.nl_count)))")

# ps = (W = [ Wˡ ∈ R^{K × n_l} per distinct l ],)  — the Stage-2 mixing weights.
initialparameters(rng::AbstractRNG, l::EquivLinearL) =
      (W = [ l.init(rng, l.rank, n) for n in l.nl_count ],)

initialstates(rng::AbstractRNG, l::EquivLinearL) = NamedTuple()

(l::EquivLinearL)(A, ps, st) = _eql_apply(l, A, ps.W), st


# Forward pass, in array form. Indices:
#   node          : a row of A (one pooled input / node) — the vectorised axis;
#   k  = 1 … K    : an output mixed channel;
#   q  = 1 … nĀ   : an output slot, i.e. a distinct (l, m); Āspec[q] = (n=1, l, m);
#   il = mix_l[q] : the W-block index for slot q = the position of its degree l in
#                   the sorted distinct l's (one block per l, so all (l,m) with
#                   the same l share il);
#   i  = 1 … n_l  : a radial channel at this l. It indexes BOTH the W-block column
#                   (Wq[k, i] = weight of the i-th radial channel into channel k)
#                   AND the gather list (cols[i] = the column of A holding that
#                   radial channel's (n, l, m) feature).
# With  cols = mix_Acols[q]  (the n_l columns of A for slot q) and  Wq = W[il]
# (size K × n_l):
#
#     Ā[node, k, q] = sum( Wq[k, i] * A[node, cols[i]]  for i = 1:n_l )
#
# i.e. mixed channel Ā[:, k, q] is the Wq[k, :]-weighted sum over the radial
# channels — the columns of A sharing slot q's (l, m). The loop below vectorises
# over the node axis (the `.+=` on A[:, cols[i]]) and accumulates one (k, i) term
# at a time. Output Ā is (nnodes, K, nĀ) with nĀ = length(Āspec).
function _eql_apply(l::EquivLinearL, A::AbstractMatrix, W)
   nnodes = size(A, 1)
   K = l.rank
   len = length(l.Āspec)
   T = promote_type(eltype(A), eltype(eltype(W)))
   Ā = zeros(T, nnodes, K, len)
   @inbounds for q = 1:len
      Wil = W[l.mix_l[q]]
      cols = l.mix_Acols[q]
      for (i, p) in enumerate(cols), k = 1:K
         w = Wil[k, i]
         @views Ā[:, k, q] .+= w .* A[:, p]
      end
   end
   return Ā
end


function _eql_pullback(∂Ā, l::EquivLinearL, A::AbstractMatrix, W)
   K = l.rank
   T = promote_type(eltype(∂Ā), eltype(A), eltype(eltype(W)))
   ∂A = zeros(T, size(A))
   ∂W = [ zeros(T, size(w)) for w in W ]
   @inbounds for q = 1:length(l.Āspec)
      il = l.mix_l[q]
      Wil = W[il]; ∂Wil = ∂W[il]
      cols = l.mix_Acols[q]
      for (i, p) in enumerate(cols), k = 1:K
         @views ∂A[:, p] .+= Wil[k, i] .* ∂Ā[:, k, q]
         ∂Wil[k, i] += dot(view(∂Ā, :, k, q), view(A, :, p))
      end
   end
   return ∂A, ∂W
end


function rrule(::typeof(_eql_apply), l::EquivLinearL, A::AbstractMatrix, W)
   Ā = _eql_apply(l, A, W)
   function _pb(∂Ā)
      ∂A, ∂W = _eql_pullback(unthunk(∂Ā), l, A, W)
      return NoTangent(), NoTangent(), ∂A, ∂W
   end
   return Ā, _pb
end
