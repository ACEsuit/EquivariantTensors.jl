using SparseArrays
using LinearAlgebra: norm 

"""
   symmetrisation_matrix(L, mb_spec; prune, kwargs...) -> 𝔸2𝔹, 𝔸_spec

Generates the symmetrization operator for a sparse ACE basis. The basis is 
# specified via the input `mb_spec`, which is a list of basis function 
specifications of the form 
```julia 
mb_spec = [ [(n=0, l=0), (n=1, l=0)], [(n=1, l=1), (n=2, l=1)], ... ] 
```
i.e. a `Vector{Vector{NL}}`. where `NL = @NamedTuple{n::Int, l::Int}`.

The parameter `L` determines the order of the ouput, e.g. L=0 for an invariant 
scalar, L = 1 for a vector, and so forth. 

The output is given in terms of a sparse matrix `𝔸2𝔹` in CCS format and a 
specification of the `𝔸` basis as a `Vector{Vector{NLM}}` where 
`NLM = @NamedTuple{n::Int, l::Int, m::Int}`. 
"""
function symmetrisation_matrix(L::Integer, mb_spec; 
                               prune = false, kwargs...)

   # for now assume a specific form of the mb_spec, namely 
   #   Vector{Vector{NT_NLM}}   
   #   where NT_NLM = typeof( (n = 0, l = 0) )

   # generate a first naive 𝔸 specification that doesn't take into account 
   # any symmetries at all. 
   #   TODO: this should be shifted into the symmetrisation operator constructor
   #
   # 
   # NOTE: this is not efficient and could be done on the fly while generating 
   #       the symmetrization operator. But for now it works and is easy to use.
   #
   # Vector{Vector{NT_NLM}}
   𝔸spec = _auto_nnllmm_spec(mb_spec)
   
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
   # (we aren't using `invmap` to avoid an intermediate allocation needed to 
   #  transform from 𝔸spec to unique keys)
   inv_𝔸spec = Dict( _bb_key(bb) => i for (i, bb) in enumerate(𝔸spec) )

   # extract all unique (nn, ll) blocks, since the (ll, mm) will only be used 
   # in generating the coupled / symmetrized basis functions
   nnll = unique( [(nn, ll) for (nn, ll, mm) in _vecnt2nnllmm.(𝔸spec)] )

   # Now for each (nn, ll) block we can generate all possible invariant basis 
   # functions. We assemble the symmetrization operator in triplet format, 
   # which can conveniently account for double-counting of entries.

   # NB : HACK TO DISTINGUISH L = 0 and L > 0 
   TVAL = L == 0 ? Float64 : SVector{2*L+1, Float64}
   irow = Int[]; jcol = Int[]; val = TVAL[]

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

   # assemble the symmetrization operator in compressed column format 
   symm = sparse(irow, jcol, val, num𝔹, length(𝔸spec)) 

   # prune rows with all-zero entries (if there are any then print a warning 
   # because this indicates a bug in `coupling_coeffs`)
   i_nz_rows = findall(!iszero, sum(norm, symm; dims = 2)[:])
   if length(i_nz_rows) != num𝔹
      @warn("symmetrization matrix has all-zero rows; this indicates a bug in `coupling_coeffs`")
      symm = symm[i_nz_rows, :]
   end

   if prune 
      i_nz_cols = sort( findall(!iszero, sum(norm, symm; dims = 1)[:]) ) 
      𝔸spec_pruned = 𝔸spec[i_nz_cols]
      symm_pruned = symm[:, i_nz_cols]
   else
      𝔸spec_pruned = 𝔸spec 
      symm_pruned = symm
   end 

   return symm_pruned, 𝔸spec_pruned
end
