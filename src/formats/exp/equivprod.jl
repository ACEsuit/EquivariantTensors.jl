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
struct EquivariantTensorProduct{NL} # make it a Lux container layer
   prodspec::Vector{Tuple{Int, Int}}
   ranges::NTuple{NL, Vector{Int}}  # ranges[L+1] = indices of the L output in the output tuple 
   LL::NTuple{NL, Int} 
   # ---- 
   meta::Dict{String, Any}
end

function EquivariantTensorProduct(LL, Rnl_spec, Ylm_spec)
   LL = tuple(LL...)

   # 1. create the full ϕ_nlm spec (all matching products)
   #    in nametuple (n =, l = , m = ) format
   #    if the radial spec carries an `l`, then a radial (n,l) only matches an
   #    angular (l',m) when l == l'; otherwise every (n, l, m) is admissible.
   radial_has_l = (:l in fieldnames(eltype(Rnl_spec)))
   ϕ_nlm_spec = NamedTuple{(:n, :l, :m), NTuple{3, Int}}[]
   for r in Rnl_spec, y in Ylm_spec
      (radial_has_l && r.l != y.l) && continue
      push!(ϕ_nlm_spec, (n = r.n, l = y.l, m = y.m))
   end
   sort!(ϕ_nlm_spec, by = b -> (b.l, b.n, b.m))

   # 2. convert ϕ_nlm spec into a prodspec::Vector{Tuple{Int, Int}}
   #    each entry is a pair (radial index, angular index) into Rnl / Ylm
   prodspec = _make_idx_A_spec(ϕ_nlm_spec, Rnl_spec, Ylm_spec)

   # 3. select the indices of the products that match the requested LL
   #    and write them into the ranges tuple
   ranges = ntuple(i -> findall(b -> b.l == LL[i], ϕ_nlm_spec), length(LL))

   # 4. store Rnl_spec, Ylm_spec, ϕ_nlm_spec,  and LL in the meta dictionary
   meta = Dict{String, Any}("Rnl_spec" => Rnl_spec,
                            "Ylm_spec" => Ylm_spec,
                            "nlm_spec" => ϕ_nlm_spec,
                            "LL" => LL)

   return EquivariantTensorProduct(prodspec, ranges, LL, meta)
end

# ------ Lux ps and st 

initialparameters(rng::AbstractRNG, bas::EquivariantTensorProduct) = 
         NamedTuple() 

initialstates(rng::AbstractRNG, bas::EquivariantTensorProduct) =
         (  prodspec = bas.prodspec,
            ranges = bas.ranges,
            LL = bas.LL,
         )


# -------- evaluation kernels 

# format of Rnl, Ylm 3-tensor is determined by EdgeEmbedding
# that is the three dimensions are (j_neig, i_node, k_feat)

function evaluate(op::EquivariantTensorProduct, 
         Rnl::AbstractArray{T, 3}, Ylm::AbstractArray{T, 3}, ps, st) where {T} 
   # just dispatch this to the ka_evaluate function.          
end


function ka_evaluate(op::EquivariantTensorProduct, 
         Rnl::AbstractArray{T, 3}, Ylm::AbstractArray{T, 3}, ps, st) where {T}

   # 1. allocate the output arrays (one for each L in LL)

   # 2. for each L launch a separate kernel _ka_evaluate_L(...) 
   #    compute the entries of the output arrays directly, sketch is
   #    given below. sync after launching all kernels. 

   # return 𝔹 a tuple of feature vectors   
end


@kernel function _ka_evaluate_L!(::Type{EquivariantTensorProduct}, 
         𝔹L,          # abstractvector SVector{2*L+1, T}, to write into 
         Rnl, Ylm,    # const abstractvector T
         prodspec,    # const abstractvector Int
         range,       # const abstractvector Int
         L::Int)
   # get the j_neig, i_node, k_feat indices, these go over the 
   # dimensions of 𝔹 
   
   # the k_feat index points to range[k_feat] which gives an index of 
   # prodspec, which is a pair of indices iR into Rnl and iY into Ylm; 
   # here we have a bug it seems. it should be a single index into Rnl and 2L+1 
   # indices into Ylm (m = -L, ..., L). Fix this in the construction of 
   # the prodspec: provide not an index iY into Ylm but only the l value
   # then use the sphericart (l, m) -> index into Ylm mapping to 
   # extract all (Ylm)_{l, m} for m = -l, ..., l as an SVector. (best 
   # with a new generated function). -> yL_vec

   # produce the Rnl[iR] * yL_vec and write it into 𝔹L[j_neig, i_node, k_feat]

end
