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
using Random: AbstractRNG


"""
   struct CPACElayer

CP / TRACE readout layer: wraps a `CPACEbasis` and contracts its per-rank carrier
features with the Stage-3b CP coefficients `λ`.

### Fields (configuration)
- `basis`    : the wrapped `CPACEbasis` (owns the Stage-2 mixing `W`).
- `nfeatures`: `NTuple` of readout counts, one per output `L` (matches
  `basis.LL`).
- `init`     : initialiser for `λ`, any `(rng, dims...) -> AbstractArray`.

### Lux parameters (`ps`) and states (`st`)
- `ps.basis` : the basis parameters — `(; W = [ Wˡ ∈ R^{K × n_l} ])`, the
  Stage-2 mixing weights (one block per distinct `l`).
- `ps.λ`     : an `NTuple` over output `L`; `ps.λ[i] ∈ R^{(K·#η_i) × nfeatures[i]}`
  are the CP coefficients `λ_{kη}` (the `(k, η)` axes flattened), contracted as
  `F_{L,feat} = Σ_{k,η} λ^L_{feat,kη} B̃ᵏ_{η,L}`.
- `st.basis` : the basis states (carrier maps etc.); the layer has no own state.
"""
struct CPACElayer{TB, NLL, FI} <: AbstractLuxLayer
   basis::TB                       # CPACEbasis
   nfeatures::NTuple{NLL, Int}     # readouts per output L (matches basis.LL)
   init::FI                        # λ initialiser, called as init(rng, K·#η, nfeat)
end

# default λ initialiser: fan-in (K·#η_i) scaling keeps the untrained output ≈ O(1)
# given the basis' Ā ≈ O(1) init. The full Nth-root calibration (trace.md §6 /
# agents/initializers.md) is deferred.
_cpl_default_init(rng::AbstractRNG, dims::Integer...) =
      et_normal(rng, dims...; σ = inv(sqrt(dims[1])))

CPACElayer(basis, nfeatures; init = _cpl_default_init) =
      CPACElayer(basis, nfeatures, init)

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
   λ = tuple([ l.init(rng, K * lens[i], nfeats[i]) for i = 1:length(LL) ]...)
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
