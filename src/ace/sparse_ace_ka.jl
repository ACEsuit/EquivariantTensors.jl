#
# KernelAbstractions evaluation of a sparse ACE model 
# 

using LinearAlgebra: transpose 
import ChainRulesCore: rrule 

# NOTES: 
#  - Rnl and Ylm must be 3-dimensional arrays; cf. SparseProdPool for 
#    the format. 
#  - this kernel is inconsistent with the agreed-format for the 𝔹 basis 
#    which is supposed to be returned as a tuple. But for initial testing,  
#    this is ok. 

function ka_evaluate(tensor::SparseACEbasis, Rnl_3, Ylm_3, ps, st)
   # out = _ka_evaluate(tensor, Rnl_3, Ylm_3, 
   #                        st.aspec, st.aaspecs, st.A2Bmaps)
   𝔹, A, 𝔸 = _ka_evaluate(tensor, Rnl_3, Ylm_3, 
                          st.aspec, st.aaspecs, st.A2Bmaps)
   return 𝔹, st 
end                           

function _ka_evaluate(tensor::SparseACEbasis, Rnl_3, Ylm_3, 
                      aspec, aaspecs, A2Bmaps)
   # A = #nodes x #features
   A = ka_evaluate(tensor.abasis, (Rnl_3, Ylm_3), aspec)
   # AA = #nodes x #features 
   AA = ka_evaluate(tensor.aabasis, A, aaspecs)
   # BB = #nodes x #features (TODO: undo the double-transpose!!!)
   BB = permutedims.( mul.(A2Bmaps, Ref(transpose(AA))) )
   return BB, A, AA
end 


function _ka_pullback(∂𝔹, tensor::SparseACEbasis, Rnl_3, Ylm_3, A, AA,
                      aspec, aaspecs, A2Bmaps)
   # 𝔹 is a tuple of bases, so ∂𝔹 is a tuple of tangents, which is
   # managed as a ChainRulesCore.Tangent. (usually thunked) By
   # extracting them as ∂𝔹[i] we get the tangent for the ith element
   # of the forward pass.
   #
   # Note: When coming through certain AD paths (e.g., Zygote through Lux),
   # ∂𝔹 might be a single matrix instead of a tuple when there's only one
   # output basis. We handle both cases here.

   # Each 𝔹[i] is of the following form:
   #      𝔹 = (𝒞 * 𝔸')' = 𝔸 * 𝒞'
   #      ∂𝔹 : 𝔹 = (∂𝔹 * 𝒞) : 𝔸
   #  =>  ∇_𝔸 (∂𝔹 : 𝔹) = ∂𝔹 * 𝒞

   # Handle both tuple and non-tuple inputs for ∂𝔹
   # - Tuple: standard case, index directly
   # - Vector of matrices: sometimes Zygote wraps tuple tangents as vectors
   # - Single matrix: when there's only one output basis, Zygote may unwrap
   # - Tangent: ChainRulesCore wraps tuple gradients in Tangent type
   _get_∂𝔹(∂𝔹::Tuple, i) = ∂𝔹[i]
   _get_∂𝔹(∂𝔹::AbstractVector{<:AbstractMatrix}, i) = ∂𝔹[i]
   _get_∂𝔹(∂𝔹::AbstractMatrix, i) = (i == 1 ? ∂𝔹 : throw(BoundsError(∂𝔹, i)))
   _get_∂𝔹(∂𝔹::ChainRulesCore.Tangent, i) = ∂𝔹[i]

   ∂𝔸 = sum( mul(_get_∂𝔹(∂𝔹, i), A2Bmaps[i], (a, b) -> sum(a .* b)) for i = 1:length(A2Bmaps) )
   ∂A = ka_pullback(∂𝔸, tensor.aabasis, A, aaspecs)
   ∂Rnl, ∂Ylm = ka_pullback(∂A, tensor.abasis, (Rnl_3, Ylm_3), aspec)
   return ∂Rnl, ∂Ylm
end 




#
# this rrule is just a wrapper for _ka_pullback
#
function rrule(::typeof(_ka_evaluate), tensor::SparseACEbasis, 
               Rnl_3, Ylm_3, aspec, aaspecs, A2Bmaps)
   𝔹, A, 𝔸 = _ka_evaluate(tensor, Rnl_3, Ylm_3, aspec, aaspecs, A2Bmaps)

   function _pb(∂𝔹A𝔸)
      ∂𝔹 = ∂𝔹A𝔸[1]
      # ∂𝔹A𝔸[2] == ∂𝔹A𝔸[2] == ZeroTangent() because A and 𝔸 are just 
      # intermediates that we keep to accelerate the backprop, but are not 
      # actually returned! 
      if !(∂𝔹A𝔸[2] == ∂𝔹A𝔸[3] == ZeroTangent())
         error("rrule for _ka_evaluate requires that only ∂𝔹 ≠ 0")
      end

      ∂Rnl, ∂Ylm = _ka_pullback(∂𝔹, tensor, Rnl_3, Ylm_3, A, 𝔸, 
                                aspec, aaspecs, A2Bmaps)
      return (∂Rnl, ∂Ylm, )
   end

   return (𝔹, A, 𝔸), ∂𝔹A𝔸 -> (NoTangent(), NoTangent(), _pb(∂𝔹A𝔸)..., 
                              NoTangent(), NoTangent(), NoTangent())
end
