

"""
   struct SelectLinL <: AbstractLuxLayer

A Lux layer which acts as a simple linear layer, but using a categorical 
variable to select a weight matrix:
```
   P -> W[x] * P 
```
This layer is experimental and likely very inefficient. 
"""
struct SelectLinL{TSEL} <: AbstractLuxLayer
   in_dim::Int
   out_dim::Int
   ncat::Int
   selector::TSEL
end

LuxCore.initialstates(rng::AbstractRNG, l::SelectLinL) = NamedTuple()

function LuxCore.initialparameters(rng::AbstractRNG, l::SelectLinL) 
   W = randn(rng, l.out_dim, l.in_dim, l.ncat) * sqrt(2 / (l.in_dim + l.out_dim))
   return (W = W,)
end

(l::SelectLinL)( P_X, ps, st) = 
      _apply_selectlinl(l, P_X[1], P_X[2], ps.W), st 

# TODO: this case distinction is a hack; it would be better to 
#       use types one cannot iterate over. Something to look into. 
#       and an argument to move to DecoratedParticles.jl

function _apply_selectlinl(l, P, x::Union{NamedTuple, Number, StaticArray},
                           W)
   B = (@view W[:, :, l.selector(x)]) * P
   return B
end


function _apply_selectlinl(l, P, X::AbstractArray, W)
   TB = promote_type(eltype(P), eltype(W))
   B = similar(P, TB, size(P, 1), l.out_dim)
   
   # Morally this should work, but it doesn't like the views it seems?!
   #       so we need write a kernel for it; fairly straightforward for this 
   #       case but unfortuntately doesn't leverage BLAS 
   # B = reduce(vcat, transpose( (@view ps.W[:, :, l.selector(x)]) * P[i, :])
   #                  for (i, x) in enumerate(X) )

   # TODO: there was a problem applying the selector when it was type unstable
   # now that this is fixed, maybe try to go back to the above implementation?
   # that way we don't have to write a custom rrule. 

   kernel! = _ka_apply_selectlinl!(KernelAbstractions.get_backend(X))
   kernel!(B, P, X, W, l.selector; ndrange = size(B))

   return B
end


@kernel function _ka_apply_selectlinl!(B, P, X, W, selector)
   # B[iB, jB] = sum P[iB, k] * W[jB, k, selector(x)]
   iB, jB = @index(Global, NTuple)
   i_x = selector(X[iB])
   B[iB, jB] = 0
   for k = 1:size(P, 2) 
      B[iB, jB] += W[jB, k, i_x] * P[iB, k]
   end
   nothing
end


# ------------------------------------------------------------
#  pullback and rrule 
#
# TODO: write tests for this 
#       these are likely wrong when multi-threaded or with GPU 
#       since they write asynchronously to ∂W; need reordering of 
#       operations or atomic adds

function _pullback_selectlinl(∂B, l, P, X::AbstractArray, W) 
   TB = promote_type(eltype(∂B), eltype(W))
   ∂P = similar(P, TB, size(P))
   ∂W = similar(W, TB, size(W))
   fill!(∂P, zero(TB))
   fill!(∂W, zero(TB))

   kernel! = _ka_pullback_selectlinl!(KernelAbstractions.get_backend(X))

   # P : nbatch x nfeat 
   # W : nout x nfeat(in) x ncategories 
   kernel!(∂P, ∂W, ∂B, P, X, W, l.selector; ndrange = size(P))

   return ∂P, ∂W
end

@kernel function _ka_pullback_selectlinl!(∂P, ∂W, ∂B, P, X, W, selector)
   # iX indexes into the "batch" and iP into the P features 
   iX, iP = @index(Global, NTuple)
   # i_x selects the category of the input X[iX] 
   i_x = selector(X[iX])

   for iout = 1:size(W, 1)
      # B[iX, iout] = ∑_k W[iout, k, i_x] * P[iX, k]    // k ≡ iP 
      ∂P[iX, iP] += W[iout, iP, i_x] * ∂B[iX, iout]
      ∂W[iout, iP, i_x] += P[iX, iP] * ∂B[iX, iout]
   end
   nothing
end

import ChainRulesCore: rrule, NoTangent, unthunk

function rrule(::typeof(_apply_selectlinl), l, 
               P::AbstractMatrix, X::AbstractArray, W::AbstractArray)

   B = _apply_selectlinl(l, P, X, W)

   function _pb_selectlinl(∂B)
      ∂P, ∂W = _pullback_selectlinl(unthunk(∂B), l, P, X, W)
      return NoTangent(), NoTangent(), ∂P, NoTangent(), ∂W
   end

   return B, _pb_selectlinl
end

function rrule(::typeof(_apply_selectlinl), l::SelectLinL, P::Tuple{AbstractMatrix},
               X::AbstractArray, W::AbstractArray)
   P_mat = P[1]
   result, pb = rrule(_apply_selectlinl, l, P_mat, X, W)
   function _pullback_tuple_wrapper(Δ)
      _, _, ∂P_mat, _, ∂W = pb(Δ)
      # Wrap ∂P back in a tuple to match input structure
      ∂P = (∂P_mat,)
      return NoTangent(), NoTangent(), ∂P, NoTangent(), ∂W
   end
   return result, _pullback_tuple_wrapper
end


# ------------------------------------------------------------
# for integration with evaluate_ed we need the following: 

# Prototype just to get things up and running. Obviously this needs to 
# be put into KernelAbstractions. 
#
function pfwd_ed(l::SelectLinL, P_dP_X, ps, st)
   P, dP, X = P_dP_X

   TB = promote_type(eltype(P), eltype(ps.W))
   dTB = promote_type(eltype(dP), eltype(ps.W))
   B = similar(P, TB, size(P, 1), l.out_dim)
   dB = similar(dP, dTB, size(dP, 1), l.out_dim)

   for i = 1:size(P, 1)
      xi = X[i]
      Wi = @view ps.W[:, :, l.selector(xi)]
      B[i, :] = Wi * P[i, :]
      dB[i, :] = Wi * dP[i, :]
   end

   return (B, dB), st 
end