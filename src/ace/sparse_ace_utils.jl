
"""
   function build_sparse_ace()
"""
function sparse_equivariant_tensor(; 
                  mb_spec, 
                  Rnl_spec = _auto_Rnl_spec(mb_spec), 
                  basis = real)
   # check that the radial spec is compatible with the mb_spec                   
   min_Rnl_spec = _auto_Rnl_spec(mb_spec)
   if !(min_Rnl_spec ⊆ Rnl_spec)
      error("mb_spec contains 1p basis functions that are not contained in Rnl_spec")
   end

   
                     
   
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
   nl_set = Set{TN}()
   for bb in mb_spec, b in bb 
      push!(nl_set, (n = b.n, l = b.l))
   end
   return sort(collect(nl_set))
end


# can we ignore the level function here? 
function _make_A_spec(AA_spec, level)
   NT_NLM = NamedTuple{(:n, :l, :m), Tuple{Int, Int, Int}}
   A_spec = NT_NLM[]
   for bb in AA_spec 
      append!(A_spec, bb)
   end
   A_spec = unique(A_spec)
   A_spec_level = [ level(b) for b in A_spec ]
   p = sortperm(A_spec_level)
   A_spec = A_spec[p]
   return A_spec
end 

# TODO: this should go into sphericart or P4ML 
function _make_Y_spec(maxl::Integer)
   NT_LM = NamedTuple{(:l, :m), Tuple{Int, Int}}
   y_spec = NT_LM[] 
   for i = 1:P4ML.SpheriCart.sizeY(maxl)
      l, m = P4ML.SpheriCart.idx2lm(i)
      push!(y_spec, (l = l, m = m))
   end
   return y_spec 
end

function _make_idx_A_spec(A_spec, r_spec, y_spec)
   inv_r_spec = _inv_list(r_spec)
   inv_y_spec = _inv_list(y_spec)
   A_spec_idx = [ (inv_r_spec[(n=b.n, l=b.l)], inv_y_spec[(l=b.l, m=b.m)]) 
                  for b in A_spec ]
   return A_spec_idx                  
end

function _make_idx_AA_spec(AA_spec, A_spec) 
   inv_A_spec = _inv_list(A_spec)
   AA_spec_idx = [ [ inv_A_spec[b] for b in bb ] for bb in AA_spec ]
   sort!.(AA_spec_idx)
   return AA_spec_idx
end 


function _generate_ace_model(rbasis, Ytype::Symbol, AA_spec::AbstractVector, 
                             Vref, 
                             level = TotalDegree(), 
                             pair_basis = nothing, 
                             ) 

   # # storing E0s with unit
   # model_meta = Dict{String, Any}("E0s" => deepcopy(E0s))
   model_meta = Dict{String, Any}()

   # generate the coupling coefficients 
   cgen = EquivariantModels.Rot3DCoeffs_real(0)
   AA2BB_map = EquivariantModels._rpi_A2B_matrix(cgen, AA_spec)

   # find which AA basis functions are actually used and discard the rest 
   keep_AA_idx = findall(sum(abs, AA2BB_map; dims = 1)[:] .> 0)
   AA_spec = AA_spec[keep_AA_idx]
   AA2BB_map = AA2BB_map[:, keep_AA_idx]

   # generate the corresponding A basis spec
   A_spec = _make_A_spec(AA_spec, level)

   # from the A basis we can generate the Y basis since we now know the 
   # maximum l value (though we probably already knew that from r_spec)
   maxl = maximum([ b.l for b in A_spec ])   
   ybasis = _make_Y_basis(Ytype, maxl)
   
   # now we need to take the human-readable specs and convert them into 
   # the layer-readable specs 
   r_spec = rbasis.spec
   y_spec = _make_Y_spec(maxl)

   # get the idx version of A_spec 
   A_spec_idx = _make_idx_A_spec(A_spec, r_spec, y_spec)

   # from this we can now generate the A basis layer                   
   a_basis = Polynomials4ML.PooledSparseProduct(A_spec_idx)
   a_basis.meta["A_spec"] = A_spec  #(also store the human-readable spec)

   # get the idx version of AA_spec
   AA_spec_idx = _make_idx_AA_spec(AA_spec, A_spec) 

   # from this we can now generate the AA basis layer
   aa_basis = Polynomials4ML.SparseSymmProdDAG(AA_spec_idx)
   aa_basis.meta["AA_spec"] = AA_spec  # (also store the human-readable spec)

   tensor = SparseEquivTensor(a_basis, aa_basis, AA2BB_map, 
                              Dict{String, Any}())

   return ACEModel(rbasis._i2z, rbasis, ybasis, 
                   tensor, pair_basis, Vref, 
                   model_meta )
end

# TODO: it is not entirely clear that the `level` is really needed here 
#       since it is implicitly already encoded in AA_spec. We need a 
#       function `auto_level` that generates level automagically from AA_spec.
function ace_model(rbasis, Ytype, AA_spec::AbstractVector, level, 
                   pair_basis, Vref)
   return _generate_ace_model(rbasis, Ytype, AA_spec, Vref, level, pair_basis)
end 
