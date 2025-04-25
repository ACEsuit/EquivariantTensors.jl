using SparseArrays

function symmetrisation_matrix(L::Integer, 
               𝔸spec::AbstractVector{<: Vector{<: NamedTuple}}; 
               prune = false, kwargs...)
   # for now assume a specific form of the 𝔸spec. Later we can generalize this 
   # to allow for different types of 𝔸spec formats and transform to something 
   # that is internally useful. 
   
   # convert an element of 𝔸spec to nn, ll, mm 
   function _vecnt2nnllmm(bb)
      nn = [ b.n for b in bb ]
      ll = [ b.l for b in bb ]
      mm = [ b.m for b in bb ]
      return nn, ll, mm
   end

   # convert nn, ll, mm to a search key, by lexicographical sorting 
   _bb_key(nn, ll, mm) = sort([ (n, l, m) for (n, l, m) in zip(nn, ll, mm) ])
   _bb_key(bb) = _bb_key( _vecnt2nnllmm(bb)... )

   # create a lookup into 𝔸spec 
   inv_𝔸spec = Dict( _bb_key(bb) => i for (i, bb) in enumerate(𝔸spec) )

   # extract all unique (nn, ll) blocks, since the (ll, mm) will only be used 
   # in generating the coupled / symmetrized basis functions
   nnll = unique( [(nn, ll) for (nn, ll, mm) in _vecnt2nnllmm.(𝔸spec)] )

   # Now for each (nn, ll) block we can generate all possible invariant basis 
   # functions. We assemble the symmetrization operator in triplet format, 
   # which can conveniently account for double-counting of entries.
   irow = Int[]; jcol = Int[]; val = Float64[]

   # counter for total number of invariant (or equivariant) basis functions
   num𝔹 = 0 
   for (nn, ll) in nnll
      # here the kwargs... should be PI and basis 
      cc, MM = O3.coupling_coeffs(L, ll, nn; kwargs...)
      num_b = size(cc, 1)   
      # lookup the corresponding (nn, ll, mm) in the 𝔸 specification 
      idx_𝔸 = [inv_𝔸spec[_bb_key(nn, ll, mm)] for mm in MM] 
      # add the new basis functions to the triplet format
      for q = 1:num_b 
         num𝔹 += 1
         for j = 1:length(idx_𝔸)
            push!(irow, num𝔹); push!(jcol, idx_𝔸[j]); push!(val, cc[q, j])
         end
      end
   end

   if prune 
      error("basis pruning not yet implemented")
   end

   # assemble the symmetrization operator in compressed column format 
   symm = sparse(irow, jcol, val, num𝔹, length(𝔸spec)) 

   return symm 
end
