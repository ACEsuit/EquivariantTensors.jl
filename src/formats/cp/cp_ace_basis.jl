#
# CP / TRACE equivariant tensor format — basis layer.
#
# Implements the 3-stage TRACE construction (agents/trace.md §2):
#   Stage 1  fixed CG carrier  C^{lη}          (reused from groups/symmop.jl)
#   Stage 2  Schur channel mix Āᵏ = Wᵏ·A       (this file; per-l, on the n-index)
#   Stage 3a per-rank symmetric product ∏ₜ Āᵏ  (reused SparseSymmProd)
#
# The never-form-`c` evaluation identity (trace.md §2):
#
#   F_LM = Σ_k Σ_η λ_{kη} B̃ᵏ_{η,LM},   B̃ᵏ_{η,LM} = Σ_m C^{LM}_{η,m} ∏ₜ Āᵏ_{lₜmₜ}
#
# This *basis* layer owns the Stage-2 mixing `W` and produces, per output L, the
# per-rank carrier features  B̃ᵏ_{η,L}  (shape (nnodes, K, #η_L), entries scalar
# for L=0 or SVector{2L+1} for L>0). The Stage-3b readout `λ` lives on the
# `CPACElayer` (cp_ace_layer.jl).
#
# Design record + open decisions: agents/trace.md.
#

using SparseArrays: SparseMatrixCSC, rowvals, nonzeros, nzrange
using LinearAlgebra: dot
import ChainRulesCore: NoTangent, rrule, unthunk
import LuxCore: AbstractLuxLayer, initialparameters, initialstates


struct CPACEbasis{NL, TA, TM, TAA, TSYM} <: AbstractLuxLayer
   abasis::TA            # PooledSparseProduct  (full multi-channel A_{nlm}; reused)
   mixer::TM             # EquivLinearL: Stage-2 channel mixing Āᵏ = Wˡ·A; owns
                         #   the mixed-channel spec (mixer.Āspec)
   aabasis::TAA          # SparseSymmProd       (∏ₜ Āᵏ, single mixed channel)
   A2Bmaps::TSYM         # NTuple per L         (carrier C^{lη}; reused)
   LL::NTuple{NL, Int}   # output irreps
   lens::NTuple{NL, Int} # #carrier rows (#η) per L
   rank::Int             # K
   ord::Int              # ν  (max correlation / body order)
   meta::Dict{String, Any}
end

Base.length(b::CPACEbasis) = sum(b.lens)

function Base.length(b::CPACEbasis, L::Integer)
   for (il, l) in enumerate(b.LL)
      l == L && return b.lens[il]
   end
   error("CPACEbasis has no output for L = $L")
end

Base.show(io::IO, b::CPACEbasis) =
      print(io, "CPACEbasis(LL = $(b.LL), rank = $(b.rank), ord = $(b.ord))")


# ----------------------------------------------------------------------
#  Constructor
#
# `mb_spec`, `Rnl_spec`, `Ylm_spec`, `basis` are exactly as for
# `sparse_equivariant_tensor(s)`. `rank = K` is the CP rank. The carrier is the
# *single-channel* carrier: after the Stage-2 mixing every l carries a single
# (mixed) channel, so the symmetric product treats equal-l factors as identical
# (trace.md §2 — this is not just convenient, it is the correct TRACE symmetry).

"""
   cp_equivariant_tensor(; LL, mb_spec, Rnl_spec, Ylm_spec, basis, rank,
                           init = EquivLinearL default)

Build a CP / TRACE equivariant tensor *basis* (`CPACEbasis`). Same carrier
arguments as `sparse_equivariant_tensors`; `rank` is the CP rank `K`. Returns a
`CPACEbasis` whose learnable parameter is the Stage-2 channel-mixing `W` (per-l
blocks `Wˡ ∈ R^{K × n_l}`). `init` is the weight initialiser for `W`, any
`(rng, dims...) -> AbstractArray` (e.g. `et_zeros`, `et_normal`); it is forwarded
to the `EquivLinearL` mixer. Combine with a `CPACElayer` for the Stage-3b `λ`
readout. See `agents/trace.md`.
"""
function cp_equivariant_tensor(; LL, mb_spec, Rnl_spec, Ylm_spec, basis,
                                 rank::Integer, init = _eql_default_init)
   K = rank

   # --- full (multi-channel) pooling spec: all (n,l,m), (n,l) ∈ mb_spec ---
   nl_full = sort(unique([ (n = b.n, l = b.l) for bb in mb_spec for b in bb ]))
   Aspec_full = sort(unique([ (n = nl.n, l = nl.l, m = m)
                              for nl in nl_full for m = -nl.l:nl.l ]))
   Aspec_full_raw = _make_idx_A_spec(Aspec_full, Rnl_spec, Ylm_spec)
   abasis = PooledSparseProduct(Aspec_full_raw)

   # --- single-channel carrier (reuse the sparse machinery) ---
   ls = sort(unique([ b.l for bb in mb_spec for b in bb ]))
   mb_spec_1 = unique([ sort([ (n = 1, l = b.l) for b in bb ]) for bb in mb_spec ])
   Rnl_spec_1 = [ (n = 1, l = l) for l in ls ]
   inner = sparse_equivariant_tensors(; LL = LL, mb_spec = mb_spec_1,
                  Rnl_spec = Rnl_spec_1, Ylm_spec = Ylm_spec, basis = basis)
   aabasis = inner.aabasis
   A2Bmaps = inner.A2Bmaps
   Āspec   = inner.meta["Aspec"]::Vector{@NamedTuple{n::Int, l::Int, m::Int}}

   # --- Stage-2 mixing index: flat A_{nlm} -> single mixed channel Ā_{lm} ---
   distinct_ls = sort(unique([ b.l for b in Āspec ]))
   nl_count = [ count(nl -> nl.l == l, nl_full) for l in distinct_ls ]
   inv_Afull = invmap(Aspec_full)
   mix_l = Vector{Int}(undef, length(Āspec))
   mix_Acols = Vector{Vector{Int}}(undef, length(Āspec))
   for (q, b) in enumerate(Āspec)
      il = findfirst(==(b.l), distinct_ls)
      ns = sort([ nl.n for nl in nl_full if nl.l == b.l ])
      mix_l[q] = il
      mix_Acols[q] = [ inv_Afull[(n = n, l = b.l, m = b.m)] for n in ns ]
   end

   mixer = EquivLinearL(K, nl_count, mix_l, mix_Acols, Āspec; init = init)

   LLt = tuple(LL...)
   lens = tuple([ size(A2Bmaps[i], 1) for i = 1:length(A2Bmaps) ]...)
   ord = maximum(length, mb_spec_1)

   meta = Dict{String, Any}(
            "Rnl_spec" => Rnl_spec, "Ylm_spec" => Ylm_spec,
            "Aspec_full" => Aspec_full, "Āspec" => Āspec,
            "mb_spec" => mb_spec, "LL" => LL, "rank" => K,
            "distinct_ls" => distinct_ls)

   return CPACEbasis(abasis, mixer, aabasis, A2Bmaps, LLt, lens, K, ord, meta)
end


# ----------------------------------------------------------------------
#  Lux integration

(l::CPACEbasis)(A, ps, st) = _cp_evaluate(l, A, ps.W), st

# the basis owns the Stage-2 mixing W (delegated to the EquivLinearL mixer)
initialparameters(rng::AbstractRNG, b::CPACEbasis) =
      initialparameters(rng, b.mixer)

initialstates(rng::AbstractRNG, b::CPACEbasis) = NamedTuple()


# ----------------------------------------------------------------------
#  Forward:  mix (all K) -> per-rank symmetric product -> carrier
#  output BB :: NTuple over L, BB[iL] :: (nnodes, K, #η_L)

# element type of B̃_L (scalar for L=0, SVector{2L+1} for L>0)
_cp_BL_eltype(A2Bmap, ::Type{T}) where {T} = typeof(zero(eltype(A2Bmap)) * one(T))

function _cp_evaluate(b::CPACEbasis, A::AbstractMatrix, W)
   nnodes = size(A, 1)
   K = b.rank
   T = promote_type(eltype(A), eltype(eltype(W)))
   nL = length(b.LL)

   Ā = _eql_apply(b.mixer, A, W)                # (nnodes, K, |Āspec|)
   BB = ntuple(iL -> Array{_cp_BL_eltype(b.A2Bmaps[iL], T)}(undef,
                                          nnodes, K, b.lens[iL]), nL)
   for k = 1:K
      𝔸k = evaluate(b.aabasis, Ā[:, k, :])      # (nnodes, |𝔸|)
      𝔸kt = permutedims(𝔸k)                     # (|𝔸|, nnodes)
      for iL = 1:nL
         B̃kL = permutedims(b.A2Bmaps[iL] * 𝔸kt) # (nnodes, #η_L)
         @views BB[iL][:, k, :] .= B̃kL
      end
   end
   return BB
end


# ----------------------------------------------------------------------
#  Pullback w.r.t. the input features A and the mixing parameters W.

# ∂𝔸[node,j] += Σ_η ⟨∂B̃[node,η], C[η,j]⟩   (carrier adjoint, one rank/L)
_cp_dotc(a, b) = sum(a .* b)   # scalar*scalar or SVector⋅SVector

function _cp_carrier_pb!(∂𝔸, A2Bmap::SparseMatrixCSC, ∂B̃)
   nnodes = size(∂B̃, 1)
   rows = rowvals(A2Bmap); vals = nonzeros(A2Bmap)
   @inbounds for j = 1:size(A2Bmap, 2)
      for idx in nzrange(A2Bmap, j)
         η = rows[idx]; c = vals[idx]
         for node = 1:nnodes
            ∂𝔸[node, j] += _cp_dotc(∂B̃[node, η], c)
         end
      end
   end
   return ∂𝔸
end

function _cp_pullback(∂BB, b::CPACEbasis, A::AbstractMatrix, W)
   nnodes = size(A, 1)
   K = b.rank
   nL = length(b.LL)
   T = promote_type(eltype(A), eltype(eltype(W)), eltype.(eltype.(∂BB))...)

   Ā = _eql_apply(b.mixer, A, W)                # (nnodes, K, |Āspec|)
   ∂Ā = zeros(T, size(Ā))
   n𝔸 = length(b.aabasis)

   for k = 1:K
      ∂𝔸k = zeros(T, nnodes, n𝔸)
      for iL = 1:nL
         _cp_carrier_pb!(∂𝔸k, b.A2Bmaps[iL], view(∂BB[iL], :, k, :))
      end
      ∂Āk = pullback(∂𝔸k, b.aabasis, Ā[:, k, :])  # (nnodes, |Āspec|)
      @views ∂Ā[:, k, :] .= ∂Āk
   end

   ∂A, ∂W = _eql_pullback(∂Ā, b.mixer, A, W)
   return ∂A, ∂W
end


function rrule(::typeof(_cp_evaluate), b::CPACEbasis, A::AbstractMatrix, W)
   BB = _cp_evaluate(b, A, W)
   function _pb(∂BB)
      ∂A, ∂W = _cp_pullback(unthunk.(unthunk(∂BB)), b, A, W)
      return NoTangent(), NoTangent(), ∂A, ∂W
   end
   return BB, _pb
end
