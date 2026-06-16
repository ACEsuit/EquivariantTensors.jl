#
# CP / TRACE format — readout layer (Stage 3b).
#
# Owns the CP coefficients `λ` (trace.md §5). Wraps a `CPACEbasis`, which
# produces per-rank carrier features B̃ᵏ_{η,L}; the layer contracts them with λ:
#
#   F_{L,feat} = Σ_{k,η} λ^L_{feat,k,η} B̃ᵏ_{η,L}
#
# Implemented as a per-L matrix multiply  (nnodes × K·#η_L) · (K·#η_L × nfeat),
# reusing `_tupmul` (which already differentiates Matrix{SVector}·Matrix{T}).
#

using LuxCore


struct CPACElayer{TB, NLL} <: AbstractLuxLayer
   basis::TB                       # CPACEbasis
   nfeatures::NTuple{NLL, Int}     # readouts per output L (matches basis.LL)
end

function Base.show(io::IO, l::CPACElayer)
   print(io, "CPACElayer(LL = $(l.basis.LL), rank = $(l.basis.rank), ",
             "nfeat = $(l.nfeatures))")
end

_get_NLL(l::CPACElayer) = length(l.basis.LL)


function LuxCore.initialparameters(rng::AbstractRNG, l::CPACElayer)
   LL = l.basis.LL
   lens = l.basis.lens
   K = l.basis.rank
   nfeats = l.nfeatures
   @assert length(LL) == length(nfeats) == length(lens)

   ps_basis = LuxCore.initialparameters(rng, l.basis)
   # λ[i] : (K·#η_i × nfeats[i]).  σ = 1/√(K·#η_i) keeps the untrained output
   # ≈ O(1) given the basis' Ā ≈ O(1) init. The full Nth-root calibration
   # (trace.md §6 / agents/initializers.md) is deferred.
   λ = tuple([ et_normal(rng, K * lens[i], nfeats[i];
                         σ = inv(sqrt(K * lens[i])))
               for i = 1:length(LL) ]...)
   return (basis = ps_basis, λ = λ)
end

LuxCore.initialstates(rng::AbstractRNG, l::CPACElayer) =
      (; basis = LuxCore.initialstates(rng, l.basis), )


(l::CPACElayer)(A, ps, st) = evaluate(l, A, ps, st)

function evaluate(l::CPACElayer, A, ps, st)
   BB, _ = l.basis(A, ps.basis, st.basis)   # NTuple over L: (nnodes, K, #η_L)
   # flatten the (K, #η_L) axes -> (nnodes, K·#η_L), then contract with λ
   nL = length(BB)
   BBr = ntuple(iL -> reshape(BB[iL], size(BB[iL], 1), :), nL)
   out = _tupmul(BBr, ps.λ)
   return out, st
end
