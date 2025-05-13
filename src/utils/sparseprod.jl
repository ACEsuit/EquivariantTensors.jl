
using StaticArrays

"""
`sparse_product(...)` : utility function to generate high-dimensional sparse grids
which are downsets.
All arguments are keyword arguments (with defaults):
* `NU` : maximum correlation order
* `minvv = 0` : `minvv[i] gives the minimum value for `vv[i]`
* `maxvv = Inf` : `maxvv[i] gives the minimum value for `vv[i]`
* `tup2bb = vv -> vv` :
* `admissible = _ -> false` : determines whether a tuple belongs to the downset
* `filter = _ -> true` : a callable object that returns true of tuple is to be kept and
false otherwise (whether or not it is part of the downset!) This is used, e.g.
to enfore conditions such as ∑ lₐ = even or |∑ mₐ| ≦ M
* `ordered = false` : whether only ordered tuples are produced; ordered tuples
correspond to  permutation-invariant basis functions
"""
sparse_product(; NU::Integer, 
                 admissible, 
            minvv = [0 for _=1:NU],
            maxvv = [(2^63-1) for _=1:NU],
            tup2bb = vv -> vv,
            filter = _-> true,
            ordered = false) =
      _sparse_product(tup2bb, admissible, filter, ordered,
                 SVector(minvv...), SVector(maxvv...))


                 
#  function barrier for `sparse_product`
function _sparse_product(tup2bb, admissible, filter, ordered,
                         minvv::SVector{NU, INT}, maxvv::SVector{NU, INT}
                  ) where {NU, INT <: Integer}

   lastidx = 0
   vv = @MVector zeros(INT, NU)
   for i = 1:NU; vv[i] = minvv[i]; end

   spec = SVector{NU, INT}[]
   orig_spec = SVector{NU, INT}[]

   # special trivial case - this should actually never occur :/
   # here, we just push an empty vector provided that the constant term 
   # is even allowed.
   if NU == 0
      if all(minvv .== 0) && admissible(vv) && filter(vv)
         push!(spec, SVector(vv))
      end
      return spec
   end

   while true
      # check whether the current vv tuple is admissible
      # the first condition is that its max index is small enough
      # we want to increment `curindex`, but if we've reach the maximum degree
      # then we need to move to the next index down
      isadmissible = true
      if any(vv .> maxvv)
         isadmissible = false
      else
         bb = tup2bb(vv)
         isadmissible = admissible(bb)
      end

      if isadmissible
         # ... then we add it to the stack  ...
         # (unless some filtering mechanism prevents it)
         if filter(bb)
            push!(spec, SVector(vv))
            push!(orig_spec, copy(SVector(vv)))
         end
         # ... and increment it
         lastidx = NU
         vv[lastidx] += 1
      else
         if lastidx == 0
            error("""lastidx == 0 should never occur; this means that the
                     smallest basis function is already inadmissible and therefore
                     the basis is empty.""")
         end

         # we have overshot, e.g. level(vv) > maxlevel or something like this
         # we must go back down, by decreasing the index at which we increment
         if lastidx == 1
            # if we have gone all the way down to lastindex==1 and are still
            # inadmissible then this means we are done
            break
         end
         # reset
         vv[lastidx-1] += 1
         vv[lastidx:end] .= 0
         # if ordered   # ordered tuples (permutation symmetry)
         #    vv[lastidx:end] .= vv[lastidx-1]
         # else         # unordered tuples (no permutation symmetry)
         #    vv[lastidx:end] .= 0
         # end
         lastidx -= 1
      end
   end

   if ordered
      # sanity check, to make sure all is as intended...
      # @assert all(issorted, orig_spec)
      # @assert length(unique(orig_spec)) == length(orig_spec)
      spec = unique(sort.(spec))
   end

   # here we used to remove the constant term in the past, but this should now 
   # be done via the filtering mechanism. 

   return spec
end



function sparse_nnll_set(; L = nothing, 
                           ORD::Integer, 
                           minn = 0, 
                           maxn::Integer, 
                           maxl::Integer, 
                           level, 
                           maxlevel)
   # generate a preliminary 1-particle basis spec, here we have to be very 
   # careful to sort `nl` by level so that no basis functions are missed. 
   nl = [ (n=n, l=l) for n = minn:maxn 
          for l = 0:maxl if level(SA[(n=n, l=l),]) <= maxlevel ]
   sort!(nl; by = b -> level(SA[b,]))

   # convert a list of indices into nl to a list of NamedTuples (bb)
   # vv[i] = 0 means this is ignored
   tup2bb = vv -> eltype(nl)[ nl[i] for i in vv if i > 0 ]

   if isnothing(L) 
      evenfilter = bb -> true 
   elseif L isa Integer 
      evenfilter = bb -> iseven(L + sum(b.l for b in bb; init=0))
   else 
      error("L = $L; I dont know what to do with this")
   end

   spec = sparse_product(; NU = ORD, 
      admissible = bb -> level(bb) <= maxlevel, 
      filter = bb -> ( (length(bb) > 0) && evenfilter(bb) ),
      tup2bb = tup2bb, 
      ordered = true, 
      maxvv = [ length(nl) for _=1:ORD ], ) 

   return tup2bb.(spec)      
end
