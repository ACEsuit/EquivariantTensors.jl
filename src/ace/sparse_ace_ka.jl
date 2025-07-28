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
   𝔹, A, 𝔸 = _ka_evaluate(tensor, Rnl_3, Ylm_3, 
                          st.aspec, st.aaspecs, st.A2Bmaps[1])
   return 𝔹, st 
end                           

function _ka_evaluate(tensor::SparseACEbasis, Rnl_3, Ylm_3, 
                      aspec, aaspecs, A2Bmap1)
   # A = #nodes x #features
   A = ka_evaluate(tensor.abasis, (Rnl_3, Ylm_3), aspec)
   # AA = #nodes x #features 
   AA = ka_evaluate(tensor.aabasis, A, aaspecs)
   # BB = #nodes x #features (TODO: undo the double-transpose!!!)
   BB = transpose( mul(A2Bmap1, transpose(AA)) )
   return BB, A, AA
end 


function ka_pullback(∂𝔹, tensor::SparseACEbasis, Rnl_3, Ylm_3, A, AA, ps, st) 
   # BB = 𝒞 * 𝔸' = (𝔸 * 𝒞')'  =>  ∇_𝔸 (∂BB : BB) = ∇_𝔸' Tr(𝔸 * 𝒞' * ∂BB)
   ∂𝔸 = mul(∂𝔹, st.A2Bmaps[1])
   ∂A = ka_pullback(∂𝔸, tensor.aabasis, A, st.aaspecs)
   ∂Rnl, ∂Ylm = ka_pullback(∂A, tensor.abasis, (Rnl_3, Ylm_3), st.aspec)
   return (∂Rnl, ∂Ylm), st 
end 


function rrule(::typeof(_ka_evaluate), tensor::SparseACEbasis, 
               Rnl_3, Ylm_3, ps, st)
   𝔹, A, 𝔸 = _ka_evaluate(tensor, Rnl_3, Ylm_3, 
                          st.aspec, st.aaspecs, st.A2Bmaps[1])

   return (𝔹, st), ∂𝔹 -> (NoTangent(), NoTangent(), 
                          ka_pullback(∂𝔹, tensor, A, 𝔸, ps, st)..., 
                          ZeroTangent(), NoTangent())
end
