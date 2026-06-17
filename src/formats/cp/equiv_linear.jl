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


struct EquivLinearL{FI} <: AbstractLuxLayer
   rank::Int                       # K mixed channels
   nl_count::Vector{Int}           # #radial channels n_l per distinct l
   mix_l::Vector{Int}              # for each output entry q: index into the W blocks
   mix_Acols::Vector{Vector{Int}}  # for each q: input columns (one per n)
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


# Ā :: (nnodes, K, len)   Āᵏ_{·q} = Σ_i Wˡ_{ki} A_{·, cols[i]}
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
