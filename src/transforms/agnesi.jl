

"""
   agnesi_params(pcut, pin, rin, req, rcut)

Precompute the parameters for the generalized Agnesi transform, to be used with 
`eval_agnesi`. See docs for ??? for details. 
"""
function agnesi_params(pcut::Integer, pin::Integer, 
                        rin::Real, req::Real, rcut::Real)
   @assert pcut > 0
   @assert pin > 0
   @assert pin >= pcut      
   @assert 0 <= rin < req < rcut

   # compute the parameter that maximizes the slope at r = req 
   a = (-2 * pin + pcut * (-2 + 4 * pin)) / (pcut + pcut^2 + pin + pin^2)
   @assert a > 0 
   # this defines the following transform 
   #   (setting rin ≡ 0, p = pcut, q = pin ... old notation)
   # x(r) = \frac{1}{1 + a (r/req)^q / (1 + (r/req)^(q-p))}
   # x(r) ~ (1+ a (r/req)^p)^(-1) ~ (r/req)^{-p} as r -> ∞ 
   # x(r) ~ (1 + a (r/req)^q)^(-1) ~ 1 - a (r/req)^q as r -> 0
   _s(r) = (r - rin) / (req - rin) 
   _x(r) = ( s = _s(r); 
             1 / (1 + a * s^pin / (1 + s^(pin - pcut))) )

   # now we want to normalize this to map to [-1, 1]
   # linear mapping to [-1, 1]
   #   x -> y(r) = 2 * (x - xcut) / (xin - xcut) - 1             
   #             = b1 * x + b0
   #  where b1 = 2 / (xin - xcut), b0 = -1 - 2 * xcut / (xin - xcut)
   xin = _x(rin)
   xcut = _x(rcut)
   b1 = 2 / (xin - xcut)
   b0 = -1 - 2 * xcut / (xin - xcut)

   # parameter structure / named-tuple 
   params = (pin = pin, pcut = pcut, 
             a = a, b0 = b0, b1 = b1, 
             rin = rin, req = req)
   
   return params 
end


""" 
   eval_agnesi(r::Real, params)

Evaluate the generalized Agnesi transform at distance r, with parameters 
provided produced by `agnesi_params`. 
"""
function eval_agnesi(r::Real, params::NamedTuple) 
   pin, pcut, a, b0, b1, rin, req = 
         params.pin, params.pcut, params.a, params.b0, params.b1, 
         params.rin, params.req

   s = (r - rin) / (req - rin)
   x = 1 / (1 + a * s^(pin) / (1 + s^(pin - pcut)))
   y = max(-1, min(1, b1 * x + b0))
   return y 
end



@doc raw"""
   agnesi_transform(...)

Construct a generalized Agnesi transform, a layer that implements the operation 
```math
   y(x) = b_0 + \frac{b_1}{1 + a s^q / (1 + s^(q-p))}
```
where 
```math
   s = \frac{r - r_{\rm in}}{r_{\rm eq} - r_{\rm in}}, 
```
with default `b_0, b_1` such that $r \in [r_{\rm in}, r_{\rm cut}]$ is mapped 
to `[-1, 1]` and `a` such that $|dy/dr|$ is maximised at $r = r_{\rm eq}$. 

The transform is constructed to satisfy (assum $r_{\rm in} = 0$)
```math 
   y \sim \frac{1}{1 + a (r/r_{\rm eq})^q} \quad \text{as} \quad r \to 0 
   \quad \text{and} 
   \quad 
   y \sim (r/r_{\rm eq})^{-p}  \quad \text{as} r \to \infty.
```

The values for $p, q, r_{\rm in}, r_{\rm eq}, r_{\rm cut}$ need to be 
   specified. Typical defaults for $p,q$ are `p = 2, q = 4`. 
"""
function agnesi_transform(categories, rins, reqs, rcuts, 
                          pin::Integer, pcut::Integer)

   function _dict_from_num(cats, r, rdef) 
      if r isa Dict 
         return r 
      end 
      if r == nothing 
         r = rdef 
      end 
      if r == nothing 
         error("cannot have r == rdef == nothing")
      end

      rd = Dict{Tuple, Real}() 
      for (c1, c2) in Iterators.product(cats, cats)
         rd[(c1, c2)] = r
      end
      return rd 
   end 

   # defaults for rins 
   rins = _dict_from_num(categories, rins, 0.0)
   # default for reqs 
   reqs = _dict_from_num(categories, reqs, nothing)
   # default for rcuts
   rcuts = _dict_from_num(categories, rcuts, nothing)

   # create the parameters 
   #     upper-triangular storage, flattened into a vector 
   #     use catcat2idx_sym to access 
   NZ = length(categories)
   params = Vector{Any}(undef, (NZ * (NZ+1)) ÷ 2)  
   idx = 0 
   for i in 1:NZ, j in i:NZ 
      # confirm that the indexing is correct
      idx += 1
      @assert idx == symidx(i, j, NZ)  
      # build the parameters for zi, zj 
      zi = categories[i]; zj = categories[j]
      params[idx] = agnesi_params(pcut, pin, 
                           rins[zi, zj], reqs[zi, zj], rcuts[zi, zj])
   end

   params = identity.(params)
   # the params should be all of the same type so can be stored in an 
   # SVector for efficiency. (is this efficient??)
   # this will be the reference state for the NTtransform 
   st = ( zlist = categories, 
          params = SVector{length(params)}(params), )

   # build the actual transform mapping 
   #     TODO: allow specification of how to get r, s0, s1 from input x!!!
   f_agnesi = let 
      (x, st) -> begin
         r = norm(x.𝐫)
         idx = catcat2idx_sym(st.zlist, x.s0, x.s1)
         return eval_agnesi(r, st.params[idx])
      end   
   end
    
   return NTtransformST(f_agnesi, st; sym = :GeneralizedAgnesi)
end
