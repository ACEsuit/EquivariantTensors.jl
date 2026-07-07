#
# implementation of an equivariant tensor product
# - should behave exactly like an equivariant linear ACE basis: 
#   input (Rnl, Ylm) => output (B0, B1, ...) 
# - 

"""
   EquivariantTensorProduct

Behaves exactly like an equivariant linear ACE basis, but without the pooling. 
Takes as input a tuple of (R, Y) edge embeddings and produces an output that 
is equivalent to an equivariant ACE basis.
"""
struct EquivariantTensorProduct # make it a Lux container layer 
   prodspec::Vector{Tuple{Int, Int}}
   ranges::NTuple{NL, Vector{Int}}  # ranges[L+1] = indices of the L output in the output tuple 
   LL::NTuple{NL, Int} 
   # ---- 
   meta::Dict{String, Any}
end

function EquivariantTensorProduct(LL, Rnl_spec, Ylm_spec) 
   # 1. create the full ϕ_nlm spec (all matching products)
   #    in nametuple (n =, l = , m = ) format 

   # 2. convert ϕ_nlm spec into a prodspec::Vector{Tuple{Int, Int}} 

   # 3. select the indices of the products that match the requested LL 
   #    and write them into the ranges tuple

   # 4. store Rnl_spec, Ylm_spec, ϕ_nlm_spec,  and LL in the meta dictionary

   # return the EquivariantTensorProduct object
end


# ------ Claude: don't edit below here

function evaluate(op::EquivariantTensorProduct, Rnl, Ylm, ps, st)
   # ϕ is a tuple of (R, Y) features 
   # ps is a tuple of (l, l') angular momentum indices 
   # st is a tuple of (s, s') spin indices 
   error("not implemented")
end


