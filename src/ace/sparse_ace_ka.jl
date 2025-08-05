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
   # BB = 𝒞 * 𝔸' = (𝔸 * 𝒞')'  =>  ∇_𝔸 (∂BB : BB) = ∇_𝔸' Tr(𝔸 * 𝒞' * ∂BB)
   # @show typeof(∂𝔹)
   @show typeof(A2Bmaps[1])
   ∂𝔸 = mul(unthunk(∂𝔹)[1], A2Bmaps[1])
   ∂A = ka_pullback(∂𝔸, tensor.aabasis, A, aaspecs)
   ∂Rnl, ∂Ylm = ka_pullback(∂A, tensor.abasis, (Rnl_3, Ylm_3), aspec)
   return ∂Rnl, ∂Ylm
end 





function rrule(::typeof(_ka_evaluate), tensor::SparseACEbasis, 
               Rnl_3, Ylm_3, aspec, aaspecs, A2Bmaps)
   𝔹, A, 𝔸 = _ka_evaluate(tensor, Rnl_3, Ylm_3, aspec, aaspecs, A2Bmaps)

   function _pb(∂𝔹A𝔸)
      ∂𝔹 = ∂𝔹A𝔸[1] 
      @show "blurg"
      ∂Rnl, ∂Ylm = _ka_pullback(∂𝔹, tensor, Rnl_3, Ylm_3, A, 𝔸, 
                                aspec, aaspecs, A2Bmaps)
      return (∂Rnl, ∂Ylm, )
   end

   return (𝔹, A, 𝔸), ∂𝔹A𝔸 -> (NoTangent(), NoTangent(), _pb(∂𝔹A𝔸)..., 
                              NoTangent(), NoTangent(), NoTangent())
end
