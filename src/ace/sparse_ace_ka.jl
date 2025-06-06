#
# KernelAbstractions evaluation of a sparse ACE model 
# 

using LinearAlgebra: transpose 

# NOTES: 
#  - Rnl and Ylm must be 3-dimensional arrays; cf. SparseProdPool for 
#    the format. 
#
function ka_evaluate(tensor::SparseACE, Rnl_3, Ylm_3, ps, st)
   A = ka_evaluate(tensor.abasis, (Rnl_3, Ylm_3), st.aspec)
   AA = ka_evaluate(tensor.aabasis, A, st.aaspecs)
   BB = mul(st.A2Bmaps[1], transpose(AA)) 
   return BB, st 
end 

# function ka_pullback(∂BB, tensor::SparseACE, A, AA, ps, st) 

# end 
