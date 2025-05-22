using SparseArrays: findnz

function sparse_equivariant_tensors(;
                  LL, 
                  mb_spec, 
                  Rnl_spec, 
                  Ylm_spec, 
                  basis, )
   A2Bmaps = [] 
   𝔸specs = [] 
   for L in LL 
      symm_L, 𝔸spec_L = symmetrisation_matrix(L, mb_spec; 
                                 prune = true, PI = true, basis = basis)
      push!(A2Bmaps, symm_L)
      push!(𝔸specs, 𝔸spec_L)                                 
   end

   # the combined 𝔸spec is just the union of all individual 𝔸specs 
   # NB: this sorting operation looks very hacky and brittle and should be 
   #     looked at very carefully; maybe one could introduce a default 
   #     ordering of the basis that is always automatically enforces and checked.
   𝔸spec = sort( union(𝔸specs...), by = bb -> (length(bb), bb) )
   inv_𝔸 = invmap(𝔸spec)

   # now we need to re-index the symmetrization operators. 
   for i = 1:length(𝔸specs)
      # map 𝔸spec_i -> 𝔸spec
      rows, cols, vals = findnz(A2Bmaps[i])
      for j = 1:length(cols) 
         bb = 𝔸specs[i][cols[j]]
         cols[j] = inv_𝔸[bb]
      end
      A2Bmaps[i] = sparse(rows, cols, vals, 
                          size(A2Bmaps[i], 1), length(𝔸spec))
   end 

   # turn the A2Bmaps into a tuple... 
   symm = tuple(A2Bmaps...)

   # now we work backwards to generate the Aspec, then the layers, 
   # see `sparse_equivariant_tensor` for for documentation of what is 
   # happening here. 
   Aspec = sort( unique( reduce(vcat, 𝔸spec) ) )
   Aspec_raw = _make_idx_A_spec(Aspec, Rnl_spec, Ylm_spec)
   𝔸spec_raw = _make_idx_AA_spec(𝔸spec, Aspec)
   Abasis = PooledSparseProduct(Aspec_raw)
   𝔸basis = SparseSymmProd(𝔸spec_raw)

   meta = Dict("Rnl_spec" => Rnl_spec, 
                "Ylm_spec" => Ylm_spec, 
                "Aspec" => Aspec, 
                "𝔸spec" => 𝔸spec, 
                "mb_spec" => mb_spec,
                "LL" => LL,)

   return SparseACE(Abasis, 𝔸basis, symm, meta)
end



"""
   sparse_equivariant_tensor(L, mb_spec, Rnl_spec, Ylm_spec, basis)
"""
function sparse_equivariant_tensor(;
                  L::Integer, 
                  mb_spec, 
                  Rnl_spec, # = _auto_Rnl_spec(mb_spec), 
                  Ylm_spec, # = _auto_Y_spec(mb_spec),
                  basis, ) # = real)
   # check that the radial spec is compatible with the mb_spec                   
   # min_Rnl_spec = _auto_Rnl_spec(mb_spec)
   # if !(min_Rnl_spec ⊆ Rnl_spec)
   #    error("mb_spec contains 1p basis functions that are not contained in Rnl_spec")
   # end

   # from this we can generate the coupling matrix and will also get a 
   # pruned 𝔸spec containing only those basis functions that are relevant 
   # for the symmetric basis 
   symm, 𝔸spec = symmetrisation_matrix(L, mb_spec; 
                                       prune = true, PI = true, basis = basis)

   # now we work backwards to generate the Aspec 
   Aspec = sort( unique( reduce(vcat, 𝔸spec) ) )
   
   # we now have the specifications for (Rnl, Ylm) -> A -> 𝔸 -> 𝔹
   # but in terms of "readable" named-tuples. We now convert these into 
   # the raw computational indices. 
   Aspec_raw = _make_idx_A_spec(Aspec, Rnl_spec, Ylm_spec)
   𝔸spec_raw = _make_idx_AA_spec(𝔸spec, Aspec)

   # now we have all information ready to generate the equivariant tensor 
   Abasis = PooledSparseProduct(Aspec_raw)
   𝔸basis = SparseSymmProd(𝔸spec_raw)
   
   meta = Dict("Rnl_spec" => Rnl_spec, 
                "Ylm_spec" => Ylm_spec, 
                "Aspec" => Aspec, 
                "𝔸spec" => 𝔸spec, 
                "mb_spec" => mb_spec,
                "L" => L,)

   return SparseACE(Abasis, 𝔸basis, (symm,), meta)                
end


"""
Takes a list of 𝔸 or 𝔹 specifications (many-body) in the form of 
```
   [  [(n=., l=., m=.), (n=., l=., m=.)], ... ]
```
and converts it into a list of sorted unique `(n=., l=.)` named pairs, 
i.e the specification of the one-body basis. 
"""
function _auto_Rnl_spec(mb_spec)
   TNL = typeof( (n = 0, l = 0) )
   nl_set = Set{TNL}()
   for bb in mb_spec, b in bb 
      push!(nl_set, (n = b.n, l = b.l))
   end
   return sort(collect(nl_set))
end

"""
   _auto_Ylm_spec(mb_spec) 

takes a list of 𝔸 or 𝔹 specifications (many-body) and return the specification 
of the Ylm basis functions as a [ (l = ., m = .), ... ] list. 
"""
function _auto_Ylm_spec(mb_spec, basis) 
   lmax = maximum(b.l for bb in mb_spec for b in bb)
   return _get_natural_Ylm_spec(lmax, basis)
end

# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # TODO: not clear this should be here? But no better idea for now...
# import Polynomials4ML as _P4ML

# _get_natural_Ylm_spec(lmax, ::typeof(real)) = 
#       _P4ML.natural_indices(_P4ML.real_sphericalharmonics(5))

# _get_natural_Ylm_spec(lmax, ::typeof(complex)) = 
#       _P4ML.natural_indices(_P4ML.complex_sphericalharmonics(5))
# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""
takes an nnll spec and generates a complete list of all possible nnllmm
"""
function _auto_nnllmm_spec(nnll_spec, Lmax = Inf)
   # NOTE: this function is a huge bottleneck of the basis generation code 
   #       but it appears that it cannot be easily improved. Using sorted 
   #       inserts is MUCH MUCH slower. Using a Set is also a little bit 
   #       slower. 
   #       Consider how to multi-thread it? 
   _sortby(bb) = (length(bb), bb) 
   NT_NLM = typeof( (n = 0, l = 0, m = 0) ) 
   nnllmm = Vector{NT_NLM}[] 
   for bb in nnll_spec
      MM = setproduct( [ -b.l:b.l for b in bb ] )
      for mm in eachrow(MM)
         # Liwei: is this really correct? 
         if abs(sum(mm)) > Lmax; continue; end 
         bb = sort!([ (n = b.n, l = b.l, m = m) for (b, m) in zip(bb, mm) ])
         push!(nnllmm, bb)
      end
   end
   sort!(nnllmm, by = _sortby)
   return unique!(nnllmm)
end


"""
convert readable A_spec into the internal representation of the A basis
"""
function _make_idx_A_spec(A_spec, 
                          r_spec::Vector{@NamedTuple{n::Int64}}, 
                          y_spec)
   inv_r_spec = invmap(r_spec)
   inv_y_spec = invmap(y_spec)
   A_spec_idx = [ (inv_r_spec[(n=b.n, )], inv_y_spec[(l=b.l, m=b.m)]) 
                  for b in A_spec ]
   return A_spec_idx                  
end

function _make_idx_A_spec(A_spec, 
                          r_spec::Vector{@NamedTuple{n::Int64, l::Int64}}, 
                          y_spec)
   inv_r_spec = invmap(r_spec)
   inv_y_spec = invmap(y_spec)
   A_spec_idx = [ (inv_r_spec[(n=b.n, l=b.l)], inv_y_spec[(l=b.l, m=b.m)]) 
                  for b in A_spec ]
   return A_spec_idx                  
end


"""
convert readable AA_spec into the internal representation of the AA basis
"""
function _make_idx_AA_spec(AA_spec, A_spec) 
   inv_A_spec = invmap(A_spec)
   AA_spec_idx = [ [ inv_A_spec[b] for b in bb ] for bb in AA_spec ]
   sort!.(AA_spec_idx)
   return AA_spec_idx
end 

