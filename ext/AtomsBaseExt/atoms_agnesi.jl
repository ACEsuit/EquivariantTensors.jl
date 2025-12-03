
# This file extends the agnesi transform to provide sensible defaults when 
# transforming an interatomic distance. 


import EquivariantTensors.Transforms: agnesi_params, eval_agnesi, 
               NTtransformST

import EquivariantTensors as ET                

function _rineqcut(zi, zj)
   req = bond_len(zi, zj)
   return (rin = req * rinfactor, req = req, rcut = req * rcutfactor)
end

function at_agnesi_params(zlist; 
                       pin = 4, pcut = 2, 
                       rcuts = nothing, 
                       rinfactor = 0.0, 
                       rcutfactor = 2.5)
   NZ = length(zlist)                           


   function __rcut(zi, zj, rcut_def) 
      if rcuts == nothing  
         return rcut_def 
      elseif rcuts isa Number 
         return rcuts
      elseif rcuts isa Dict 
         if haskey(rcuts, (zi, zj))
            return rcuts[(zi, zj)]
         elseif haskey(rcuts, (zj, zi))
            return rcuts[(zj, zi)]
         elseif haskey(rcuts, "default")
            return rcuts["default"]
         else 
            return rcut_def 
         end
      end 
      error("agnesi_params: illegal format for rcuts parameter.")
   end 

   idx = 0 
   for i in 1:NZ, j in i:NZ 
      # confirm that the indexing is correct
      idx += 1
      @assert idx == symidx(i, j, NZ)  
      # build the parameters for zi, zj 
      zi = zlist[i]; zj = zlist[j]
      rs = _rineqcut(zi, zj)
      params[idx] = agnesi_params(pcut, pin, rs.rin, rs.req, 
                                   __rcut(zi, zj, rs.rcut)) 
   end

   return identity.(params)
end


function ET.Atoms.agnesi_transform(zlist; kwargs...) 

   params = at_agnesi_params(zlist; kwargs...)

   # the params should be all of the same type so can be stored in an 
   # SVector for efficiency. (is this efficient??)
   # this will be the reference state for the NTtransform 
   st = ( zlist = zlist, 
          params = SVector{length(params)}(params), )

   # build the actual transform mapping 
   f_agnesi = let 
      (x, st) -> begin
         r = norm(x.𝐫)
         idx = catcat2idx_sym(st.zlist, x.s0, x.s1)
         y = eval_agnesi(r, st.params[idx])
         return y 
      end   
   end

   return NTtransformST(f_agnesi, st; 
                        sym = :GeneralizedAgnesi)
end
