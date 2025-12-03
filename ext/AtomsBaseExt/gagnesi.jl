

import EquivariantTensors.Transforms: gagnesi_params, eval_gagnesi, 
            NTtransformST

function rineqcut(zi, zj)
   req = bond_len(zi, zj)
   return (rin = req * rinfactor, req = req, rcut = req * rcutfactor)
end

function gagnesi_params(zlist; 
                        pin = 4, pcut = 2, 
                        rinfactor = 0.0, rcutfactor = 2.5)
   NZ = length(zlist)                           

   # upper-triangular storage, flattened into a vector 
   # use catcat2idx_sym to access 
   params = Vector{Any}(undef, (NZ * (NZ+1)) ÷ 2)  

   idx = 0 
   for i in 1:NZ, j in i:NZ 
      # confirm that the indexing is correct
      idx += 1
      @assert idx == symidx(i, j, NZ)  
      # build the parameters for zi, zj 
      rs = rineqcut(zlist[i], zlist[j])
      params[idx] = gagnesi_params(pcut, pin, rs.rin, rs.req, rs.rcut)
   end

   return identity.(params)
end


function gagnesi_transform(zlist; 
                           pin = 4, pcut = 2, 
                           rinfactor = 0.0, rcutfactor = 2.5)

   params = gagnesi_params(zlist; pin = pin, pcut = pcut, 
                           rinfactor = rinfactor, rcutfactor = rcutfactor)

   # the params should be all of the same type so can be stored in an 
   # SVector for efficiency. (is this efficient??)
   # this will be the reference state for the NTtransform 
   st = (params = SVector{length(params)}(params), )

   # build the actual transform mapping 
   f_agnesi = let zlist = deepcopy(zlist), 
      (x, st) -> begin
         r = norm(x.𝐫)
         idx = catcat2idx_sym(zlist, x.s0, x.s1)
         params = st.params[idx]
         y = eval_gagnesi(r, params)
         return y 
      end   
   end

   return NTtransformST(f_agnesi, st; sym = :GeneralizedAgnesi)
end
