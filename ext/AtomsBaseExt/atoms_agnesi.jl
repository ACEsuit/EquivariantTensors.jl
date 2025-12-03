
# This file extends the agnesi transform to provide sensible defaults when 
# transforming an interatomic distance. 


import EquivariantTensors: agnesi_params, eval_agnesi, 
                           NTtransformST

import EquivariantTensors as ET                

function _rineqcut(zi, zj, rinfactor, rcutfactor)
   req = bond_len(zi, zj)
   return (rin = req * rinfactor, req = req, rcut = req * rcutfactor)
end

function at_agnesi_params(zlist; 
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
   rins1 = Dict(); reqs1 = Dict(); rcuts1 = Dict();
   for i in 1:NZ, j in i:NZ 
      zi = zlist[i]; zj = zlist[j]
      rs = _rineqcut(zi, zj, rinfactor, rcutfactor) 
      rins1[(zi, zj)] = rs.rin
      reqs1[(zi, zj)] = rs.req
      rcuts1[(zi, zj)] = __rcut(zi, zj, rs.rcut)
   end

   return rins1, reqs1, rcuts1
end


function ET.Atoms.agnesi_transform(zlist; pin = 4, pcut = 2, kwargs...) 
   rins, reqs, rcuts = at_agnesi_params(zlist; kwargs...)

   display(rins)
   display(reqs)
   display(rcuts)

   return ET.agnesi_transform(zlist, rins, reqs, rcuts, pin, pcut)
end
